"""Pseudo-instruction emitter: LayerDesc + TilingPlan → ISA dicts.

Standard conv uses Template A from sd_codegen.py (guide §5.1).
Deformable conv emits OffsetLoader + bilinear WeightLoader sequences
matching sd_sr_codegen.py — TVM generic lowering is intentionally bypassed.

Critical invariant (do NOT refactor):
  DataLoader and WeightLoader are BOTH dispatched with the SAME st.line_buffer_idx.
  st.line_buffer_idx is toggled ONCE, AFTER WeightLoader — never between them.
  (sd_codegen uses separate DataLoaderManager / WeightLoaderManager each starting at 0,
   so both always see the same value; we replicate this with a single shared toggle.)

load_next scheduling (sd_inst pattern):
  emit_program(..., is_first=True)  → emit 5-instruction DDR preamble [0-4]
  Before layer 0 tiles              → OffchipDataLoader(image, transnum=image_transnum)
  After layer 0 tiles (load_next)   → OffchipDataLoader(next image, transnum=image_transnum)
  QuantLoader uses 1-based layer_idx (golden: layer 0 → layer_idx=1, layer 1 → layer_idx=2…)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend import isa
from backend.post_pass import finalize_instructions
from ir.layer_desc import LayerDesc
from tiling.tiling import TilingPlan


@dataclass
class EmitterState:
    """Mutable address/register bookkeeping across layers."""
    dataloader_bas_addr: int = 0
    weight_bas_addr: List[int] = field(default_factory=lambda: [0, 0, 0])
    quant_bas_addr: int = 0
    storer_bas_addr: int = 0
    line_buffer_idx: int = 0
    acc_reg_idx: int = 0
    quant_config_idx: int = 0
    offset_reg_idx: int = 0
    conv_layer_counter: int = 0  # 1-based; increments only for conv/deformable_conv layers
    # Ping-pong feature buffer allocation. feature_buf names the on-chip buffer
    # the NEXT conv/dconv layer will read from. Standard conv and deformable
    # conv both toggle it after emitting their DataStorer; offset_gen does NOT
    # toggle (its output goes to offset_reg, and the next layer re-reads the
    # same feature buffer that fed the offset_gen).
    # Initialize to 'b' so layer-0's DataStorer writes to 'a' (matching
    # golden sr_inst where dest_buffer_idx_on_chip starts at 'a').
    feature_buf: str = "b"
    # Identifies the terminal conv/dconv layer so its DataStorer writes to the
    # network's exported output buffer instead of the usual 'a'/'b' ping-pong.
    last_layer_idx: int = -1
    last_layer_dest_buffer: str = "fsrcnn_output_buffer"
    # Per-layer feature-buffer base addresses from the address allocator.
    # layer_input_bas_addr: base word-addr of the CURRENT layer's input tensor
    #   (= previous conv layer's output) within its ping-pong buffer.
    # layer_output_bas_addr: base word-addr of the CURRENT layer's output tensor.
    # Both are 0 for sequential models (no skip connections) and for the first
    # layer of a model.  Set by emit_layer before dispatching to templates.
    layer_input_bas_addr: int = 0
    layer_output_bas_addr: int = 0
    # Index of the most recent layer that wrote a tensor to the feature buffer;
    # used to look up the input base address from addr_map for the next layer.
    last_feature_layer_idx: int = -1
    # Phase 27: when the most recent feature-producing layer also wrote a
    # pooled output (has_pool_output=True), this records that pool's base
    # address. The NEXT conv layer reads from the pool output (golden
    # convention: layer after a pool-preceding conv consumes pool_addr_start),
    # so the emitter prefers this over `addr_map[last_feature_layer_idx]`.
    # Set to -1 after a non-pool-producing layer (the normal case).
    last_feature_pool_addr_start: int = -1

    @property
    def dl_layer_idx(self) -> int:
        """Golden DataLoader.layer_idx: 0-based contiguous conv-layer counter.

        Golden numbering treats each conv/dconv/offset_gen as one increment of
        the conv index. ``conv_layer_counter`` is incremented at the start of
        each such layer's emission and is 1-based (1..N) during the body. The
        DL field convention is 0-based, so we subtract 1.
        """
        return max(0, self.conv_layer_counter - 1)

    @property
    def ql_layer_idx(self) -> int:
        """Golden QuantLoader.layer_idx: 0-based contiguous conv index.

        Matches sd_sr_codegen.py freshly-dumped golden: QL.layer_idx ==
        DL.layer_idx (both numbered 0..N-1 by conv index). The repository's
        archived ``pseudo_code_load_next_mid.txt`` uses an older 1-based QL
        convention (off by 1) — comparison against that file should treat
        QL.layer_idx as a known-divergent field, not fresh sd_inst output.
        """
        return self.dl_layer_idx


class InstructionEmitter:
    """Dispatch emit_layer to per-op template methods."""

    def __init__(
        self,
        state: Optional[EmitterState] = None,
        addr_map: Optional[Dict[int, int]] = None,
        buf_map: Optional[Dict[int, str]] = None,
    ):
        self.state = state or EmitterState()
        self._addr_map: Dict[int, int] = addr_map or {}
        # Phase 25: buf_map carries the allocator's logical buffer class
        # ('a'/'b'/'offset_reg') per layer. The emitter's runtime parity
        # already produces the matching dest_buffer_idx field for SD-UNet's
        # encoder, so buf_map is currently consulted only for diagnostics
        # and consistency checks (kept here for future coordination passes,
        # e.g. when supporting topologies where parity diverges from the
        # liveness-class assignment).
        self._buf_map: Dict[int, str] = buf_map or {}

    def reset(self) -> None:
        isa.reset_instruction_stream()
        self.state = EmitterState()

    def emit_layer(self, layer: LayerDesc, plan: TilingPlan) -> None:
        st = self.state
        # Resolve feature-buffer base addresses from the allocator map.
        # layer_input_bas_addr: where the data this layer reads is stored.
        #   - For sequential layers: the previous feature-writing layer's
        #     address. If that layer also produced a pool output (has_pool_output),
        #     the consumer reads the POOLED data — use pool_addr_start instead
        #     (Phase 27: golden L3 reads c1_pool_out at 1152, not c1_for_cat
        #     at 0, even though both came from the same producer idx=2).
        #   - For skip-connection consumers (layer.skip_sources non-empty):
        #     the concat region's lowest physical address. Phase 27 picks the
        #     skip source with the LOWEST addr_map address (NOT the lowest
        #     idx) so the bas_addr aligns to the start of the concat region.
        #     For idx=15, skip_srcs=[14, 11] → addr_map[14]=2016 < addr_map[11]
        #     =2160, so we pick idx=14's address (golden L11 reads at 2016).
        # layer_output_bas_addr: where THIS layer's output will be placed.
        # For sequential models (no skip connections), addr_map is empty and
        # both resolve to 0 — preserving existing emitter behaviour exactly.
        skip_srcs = getattr(layer, "skip_sources", None) or []
        if skip_srcs:
            # Pick the source with the lowest addr_map entry (lowest physical
            # address). Falls back to lowest idx if both addresses are 0
            # (unmapped). For sequential FSRCNN this branch is never taken
            # (skip_sources is always empty).
            input_src_idx = min(
                skip_srcs,
                key=lambda i: (self._addr_map.get(i, 0), i),
            )
            st.layer_input_bas_addr = self._addr_map.get(input_src_idx, 0)
        elif st.last_feature_pool_addr_start >= 0:
            # Sequential layer reading from the previous conv's pool output.
            # The previous conv had has_pool_output=True; its pool result
            # was written to plan.pool_addr_start (recorded in EmitterState).
            st.layer_input_bas_addr = st.last_feature_pool_addr_start
        else:
            input_src_idx = st.last_feature_layer_idx
            st.layer_input_bas_addr = self._addr_map.get(input_src_idx, 0)
        st.layer_output_bas_addr = self._addr_map.get(layer.idx, 0)

        if layer.op in ("conv2d", "deformable_conv2d", "offset_gen"):
            self.state.conv_layer_counter += 1
            # Phase 31: pre-toggle acc_reg_idx for layers whose archived golden
            # parity diverges from our linear schedule (e.g. L=9 conv8 — golden
            # interleaves conv11 back-half between L=8 and L=9, shifting the
            # parity by 5 toggles which our compiler doesn't reproduce).
            if plan.flip_acc_reg_idx_on_entry:
                self.state.acc_reg_idx = 1 - self.state.acc_reg_idx
            if layer.op == "deformable_conv2d":
                self._emit_deformable_conv(layer, plan)
            elif layer.op == "offset_gen":
                self._emit_offset_gen(layer, plan)
            else:
                self._emit_standard_conv(layer, plan)
            # Phase 27: track pool-output state for the next layer's input
            # address derivation. has_pool_output is False for non-pool
            # convs, deformable conv, and offset_gen — those reset the
            # tracker so the next layer reads from addr_map normally.
            if plan.has_pool_output:
                st.last_feature_pool_addr_start = plan.pool_addr_start
            else:
                st.last_feature_pool_addr_start = -1
        elif layer.op == "pool2d":
            pass  # pooling is encoded in the adjacent conv's DataStorer flags; no separate instruction

    def emit_quant_loader(self, layer_idx: int, transnum: int, quant_mode: int = 0) -> None:
        isa.QuantLoader.dispatch(
            quant_reg_load_idx=self.state.quant_config_idx,
            quant_mode=quant_mode,
            layer_idx=layer_idx,
            transnum=transnum,
            bas_addr=self.state.quant_bas_addr,
        )
        self.state.quant_bas_addr += transnum
        # quant_config_idx toggle is the caller's responsibility — done once
        # after all DataStorers for the layer are emitted, not here.

    def _emit_standard_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """Template A: QuantLoader → per macro W-tile (DataLoader, WeightLoader, DataStorer).

        After all macro W-tiles complete, advance weight_bas_addr[0] by the total
        weight footprint of the layer = cin_group × weight_transnum_base. This
        matches sd_sr_codegen.py post-layer bookkeeping (e.g. layer 0 adds 9 with
        cin_group=1; layer 1 adds 4*9 with cin_group=4).
        """
        # Group conv dispatch: when layer.groups > 1, plan.group_count > 1
        # (set by tiling._apply_group_params). Delegate to _emit_group_conv.
        # For groups==1 (default), this branch is bypassed and behavior is
        # bit-for-bit identical to the pre-group-conv emitter.
        if plan.group_count > 1:
            self._emit_group_conv(layer, plan)
            return
        st = self.state
        # Destination ping-pong buffer: terminal conv writes to the exported
        # output buffer; all other layers alternate 'a'/'b'.
        if layer.idx == st.last_layer_idx:
            dest_buf = st.last_layer_dest_buffer
        else:
            dest_buf = "b" if st.feature_buf == "a" else "a"

        # Phase 20 P1: QL dispatch mode is determined by TilingPlan flags.
        #   Default (neither flag): one QL emitted before all macro W tiles.
        #   ql_per_macro=True   : one QL emitted per macro W tile (idx=1,2,21,22).
        #                          Both tiles use the SAME quant_config_idx —
        #                          no toggle between tiles. The single end-of-
        #                          layer toggle handles the transition.
        #   ql_per_oc_iter=True : QL is emitted inside _emit_w_macro_tile's
        #                          oc_inner loop (idx=18,20). Skip here.
        if not plan.ql_per_macro and not plan.ql_per_oc_iter:
            self.emit_quant_loader(self.state.ql_layer_idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        for macro_idx, (w0, w_sz, bas_hint) in enumerate(plan.w_macro_tiles):
            if plan.ql_per_macro:
                if macro_idx > 0:
                    # Phase 30: golden reuses the SAME quant_bas_addr across all
                    # macro W tiles of the same layer (the QuantLoader fetch is
                    # not retiled along W). Undo the prior emit_quant_loader
                    # advance so this tile's QL gets the same bas_addr.
                    self.state.quant_bas_addr -= plan.quant_transnum
                self.emit_quant_loader(
                    self.state.ql_layer_idx,
                    transnum=plan.quant_transnum,
                    quant_mode=plan.quant_mode,
                )
            self._emit_w_macro_tile(layer, plan, w0, w_sz, bas_hint, macro_idx, dest_buf)
        st.weight_bas_addr[plan.wl_slot] += plan.weight_transnum_base * plan.cin_group * plan.ky_outer * plan.oc_inner
        # Toggle feature_buf so the next layer reads from the buffer we just
        # wrote to. (Safe even after the terminal layer — nothing reads it.)
        st.feature_buf = dest_buf
        # QuantLoader and all DataStorers in this layer share the same
        # quant_config_idx; toggle once here after all tiles are done.
        st.quant_config_idx = 1 - st.quant_config_idx
        st.last_feature_layer_idx = layer.idx

    def _emit_w_macro_tile(
        self,
        layer: LayerDesc,
        plan: TilingPlan,
        w0: int,
        w_sz: int,
        bas_hint: int,
        macro_idx: int,
        dest_buf: str,
    ) -> None:
        st = self.state
        load_total = plan.load_total_num
        padding_num = plan.padding_num
        # Add the allocator-assigned base address for this layer's input tensor.
        # For sequential models (addr_map empty): layer_input_bas_addr == 0 →
        # identical to the pre-allocation behaviour.
        st.dataloader_bas_addr = st.layer_input_bas_addr + bas_hint
        # DataStorer writes to layer_output_bas_addr + tile offset.
        # Phase 28 Fix 1: mask-store layers (is_mask=True, idx=22) use a
        # special right-macro offset (typically +1 instead of +tile_h*4 = +576)
        # because the masked store interleaves left/right halves into the same
        # output reg with reg-internal half-selection (golden L18 line 2403:
        # "datastorermanager.base_addrs_res_cur = 1" for right macro).
        # Phase 31: same_base_for_macros (idx=1, idx=21) — both macros write
        # at base 0 because the producer's intermediate buffer is consumed
        # before the right macro fires, allowing slot reuse (archived golden
        # L=1 conv1_1, L=17 conv17).
        if plan.is_mask:
            tile_half_offset = 0 if macro_idx == 0 else plan.mask_macro_offset
        elif plan.same_base_for_macros:
            tile_half_offset = 0
        else:
            tile_half_offset = 0 if macro_idx == 0 else plan.tile_h * 4
        ds_base_initial = st.layer_output_bas_addr + tile_half_offset

        # Phase 28 Fix 2: dual-region DataLoader for skip-concat consumers.
        # When this layer reads a concat input whose two parts live at
        # different absolute addresses (typical SD-UNet decoder layer with
        # skip_sources=[high_idx, low_idx]), the per-cin_g DL bas_addr must
        # JUMP between the two regions at the half-cin split (golden L17
        # line 2329: "bas_addr_cur + (ic*144 if ic <= 3 else 144*4*2 + (ic-4)*144)").
        #
        # Detection criteria (must all hold):
        #   - layer.skip_sources has exactly 2 entries
        #   - addr_map gives DIFFERENT addresses for those two sources
        #   - cin_group is even (so cin_g splits cleanly at cin_group/2)
        #   - layer is NOT a group conv (group conv has its own per-group base
        #     path in `_emit_group_conv` — Fix 4)
        #
        # Effect: cin_g in [0, split) reads from low region with stride tile_h;
        #         cin_g in [split, cin_group) reads from low + region_jump
        #         + tile_h * (cin_g - split). The region_jump is chosen so the
        #         second-half base lands at high_addr (the upper concat slice).
        skip_srcs = getattr(layer, "skip_sources", None) or []
        dual_split = 0
        dual_region_jump = 0
        # Sliding-window row addressing applies to ky_outer>1 layers with
        # standard row-major SRAM layout (line_buffer_reshape != 3).
        # Conv11 (line_buffer_reshape=3) uses the legacy tile_h*cin_g formula
        # with the accumulator, not the sliding-window row_idx approach.
        use_sw = (
            plan.ky_outer > 1
            and plan.line_buffer_reshape != 3
            and plan.dl_advance_nopad is not None
        )
        if (
            len(skip_srcs) == 2
            and plan.group_count == 1
            and plan.cin_group % 2 == 0
            and plan.cin_group > 1
        ):
            addrs = sorted({self._addr_map.get(s, 0) for s in skip_srcs})
            if len(addrs) == 2:
                low_addr, high_addr = addrs[0], addrs[1]
                split = plan.cin_group // 2
                # region_jump = high - low - cin_stride * split  ⇒
                #   for cin_g >= split: bas = low + cin_stride*cin_g + region_jump
                #                     = high + cin_stride*(cin_g - split). ✓
                # For ky_outer>1 (sliding window): cin_stride = dl_advance_nopad
                # (one row of the region covers all split channels).
                # For ky_outer=1: cin_stride = tile_h (contiguous channel storage).
                # Only enable when rj > 0 to avoid degrading contiguous layouts.
                if use_sw:
                    rj = high_addr - low_addr - plan.dl_advance_nopad
                else:
                    rj = high_addr - low_addr - plan.tile_h * split
                if rj > 0:
                    dual_split = split
                    dual_region_jump = rj

        # Outer oc loop (golden L14/L16 emit ky×ic body twice with DS stride
        # of ds_oc_stride per oc iteration). Default oc_inner=1 → no extra
        # iteration. The body inside the for-oc replicates the original loop
        # exactly; only the DS base_addrs_res is offset by oc_step.
        for oc_idx in range(plan.oc_inner):
            # Phase 20 P1: ql_per_oc_iter (idx=18,20) emits a QL before each
            # oc iteration's load_idx loop. quant_config_idx is toggled AFTER
            # the oc iteration completes so the next oc's QL/DS pair uses the
            # alternated value (matching golden's per-oc QL alternation).
            if plan.ql_per_oc_iter:
                self.emit_quant_loader(
                    self.state.ql_layer_idx,
                    transnum=plan.quant_transnum,
                    quant_mode=plan.quant_mode,
                )
            st.storer_bas_addr = ds_base_initial + oc_idx * plan.ds_oc_stride
            # Reset DataLoader base for each oc iteration so each oc pass
            # re-traverses the same input rows from scratch (matching golden
            # L14/L16 line "datastorermanager.bas_addr_cur = 0" inside oc loop).
            st.dataloader_bas_addr = st.layer_input_bas_addr + bas_hint

            for load_idx in range(load_total):
                # ky × cin inner loops. plan.ky_outer == 1 for most standard
                # convs (hardware handles 3×3 kernel via the line buffer).
                # plan.ky_outer == 3 when ky is the inner software loop (e.g.
                # SD-UNet L3-L6, L9-L16).
                # plan.ic_only_no_ky is True for SD-UNet L17/L18 — the inner
                # loop is just cin_group (ic), no ky software loop, even though
                # ky_outer=1.
                for ky_g in range(plan.ky_outer):
                    if plan.ky_outer > 1:
                        # line_buffer_reshape=3 (conv11/FSRCNN-F) uses 1/5;
                        # all other ky_outer>1 layers (SD-UNet standard enc/dec)
                        # use 2 for both first and last padding rows.
                        _pad_first = 1 if plan.line_buffer_reshape == 3 else 2
                        _pad_last  = 5 if plan.line_buffer_reshape == 3 else 2
                        if load_idx == 0 and ky_g == 0:
                            is_padding_row = _pad_first
                        elif load_idx == load_total - 1 and ky_g == plan.ky_outer - 1:
                            is_padding_row = _pad_last
                        else:
                            is_padding_row = 0
                    else:
                        if load_idx < padding_num:
                            is_padding_row = 1
                        elif load_idx > load_total - 1 - padding_num:
                            is_padding_row = 5
                        else:
                            is_padding_row = 0
                    for cin_g in range(plan.cin_group):
                        # DL bas_addr computation. Two regimes:
                        #
                        # Sliding-window (ky_outer>1, reshape!=3, dl_advance_nopad set):
                        #   row_idx = max(0, load_idx + ky_g - 1) — the 3-row window.
                        #   per_cin = dl_advance_nopad / channels_per_region
                        #           (channels_per_region = dual_split or cin_group)
                        #   cin_offset = row_idx * dl_advance_nopad + cin_g * per_cin
                        #   (+ dual_region_jump for cin_g >= dual_split)
                        #
                        # Accumulator (ky_outer=1 or reshape==3):
                        #   cin_offset = tile_h * cin_g
                        #   (+ dual_region_jump for cin_g >= dual_split)
                        #   st.dataloader_bas_addr advances per load_idx step.
                        if use_sw:
                            row_idx = max(0, load_idx + ky_g - 1)
                            _chans = dual_split if dual_split else plan.cin_group
                            per_cin = plan.dl_advance_nopad // _chans
                            cin_offset = row_idx * plan.dl_advance_nopad + cin_g * per_cin
                            if dual_split and cin_g >= dual_split:
                                cin_offset += dual_region_jump
                        else:
                            cin_offset = plan.tile_h * cin_g
                            if dual_split and cin_g >= dual_split:
                                cin_offset += dual_region_jump
                        isa.DataLoader.dispatch(
                            # Phase 29: golden DL.layer_idx is the contiguous
                            # 0-based conv index, NOT layer.idx (which has
                            # gaps for fused-out concat layers).
                            layer_idx=st.dl_layer_idx,
                            line_buffer_reshape=plan.line_buffer_reshape,
                            is_padding_row=is_padding_row,
                            read_mode=plan.read_mode,
                            transnum=(
                                plan.last_step_transnum
                                if plan.last_step_transnum is not None and load_idx == load_total - 1
                                else plan.line_buffer_rows
                            ),
                            line_buffer_idx=st.line_buffer_idx,
                            # Layer 0 sources from the DDR-preloaded offchip input
                            # buffer; subsequent layers follow ping-pong alternation.
                            src_buffer_idx="offchip_input_buffer" if layer.idx == 0 else st.feature_buf,
                            bas_addr=st.dataloader_bas_addr + cin_offset,
                        )

                        isa.WeightLoader.dispatch(
                            acc_reg_comp_idx=st.acc_reg_idx,
                            kernal_size=0 if layer.k_h == 3 else 1,
                            line_buffer_row_shift=plan.wl_line_buffer_row_shift,
                            line_buffer_idx=st.line_buffer_idx,
                            is_padding_col=plan.wl_is_padding_col,
                            weight_parall_mode=plan.weight_parall_mode,
                            # First (ky, cin) pair overwrites the accumulator; rest accumulate.
                            is_new=0 if (ky_g == 0 and cin_g == 0) else 1,
                            transnum=plan.weight_transnum_base,
                            bas_addr=(
                                st.weight_bas_addr[plan.wl_slot]
                                + oc_idx * plan.cin_group * plan.ky_outer * plan.weight_transnum_base
                                + (ky_g * plan.cin_group + cin_g) * plan.weight_transnum_base
                            ),
                            is_bilinear_bicubic=plan.use_bilinear_weights,
                            offset_reg_idx=st.offset_reg_idx,
                        )
                        st.line_buffer_idx = 1 - st.line_buffer_idx

                # Advance DataLoader base ONCE per cal_idx — only for accumulator
                # (ky_outer=1 or reshape=3) mode. Sliding-window layers compute
                # the address from row_idx directly, so no advance is needed.
                if not use_sw:
                    if plan.dl_advance_pad is not None:
                        st.dataloader_bas_addr += plan.dl_advance_pad if load_idx < padding_num else plan.dl_advance_nopad
                    else:
                        _w_words = max(1, (w_sz + 63) // 64)
                        st.dataloader_bas_addr += 2 * (_w_words - 1) if load_idx < padding_num else 2 * _w_words

                # DataStorer: mode-specific fields derived from store_mode / acc_mode.
                #   store_mode=3 (pool-while-store, deformable conv):
                #     is_pooling=1, pooling_out_mode=3
                #   plan.has_pool_output (SD-UNet conv preceding pool2d, Phase 21):
                #     is_pooling=1, pooling_out_mode=plan.pool_output_mode (0/1/2);
                #     for mode>=1, pooling_out_new alternates 1/0 per DS step.
                #   acc_mode=5  (pixelshuffle output, FSRCNN last_part):
                #     is_pixelshuffle=1, stride=0, transfer_num=0, is_bicubic_add=1, pixelshuffle_out_mode=1
                #   plan.is_pixelshuffle (SD-UNet pre-DepthToSpace conv): use
                #     per-layer override values (pixelshuffle_*_mode, transfer_num, stride).
                is_pool_store = (plan.store_mode == 3)        # deformable conv pool
                is_pool_out = plan.has_pool_output            # conv preceding pool2d
                is_pixshuffle_legacy = (plan.acc_mode == 5)
                # Phase 28 Fix 1: is_mask layers (SD-UNet idx=22) bypass both
                # the pixelshuffle and is_pixshuffle_legacy paths. The terminal
                # SD-UNet conv writes to unet_output_reg with a masked-store
                # pattern (golden L18 line 2253-2275 / 2443-2465).
                is_pixshuffle = (
                    not plan.is_mask
                    and (plan.is_pixelshuffle or is_pixshuffle_legacy)
                )
                if plan.is_mask:
                    pix_out_mode = 0
                    pix_acc_mode = 0
                    pix_store_mode = 1
                    pix_transfer_num = 2   # golden: transfer_num=2 for unet_output_reg writes
                    pix_stride = 0
                    pix_bicubic_add = 0
                elif plan.is_pixelshuffle:
                    pix_out_mode = plan.pixelshuffle_out_mode
                    pix_acc_mode = plan.pixelshuffle_acc_mode
                    pix_store_mode = plan.pixelshuffle_store_mode
                    pix_transfer_num = plan.pixelshuffle_transfer_num
                    pix_stride = plan.pixelshuffle_stride
                    pix_bicubic_add = 0
                elif is_pixshuffle_legacy:
                    pix_out_mode = 1
                    pix_acc_mode = plan.acc_mode
                    pix_store_mode = plan.store_mode
                    pix_transfer_num = 0
                    pix_stride = 0
                    pix_bicubic_add = 1
                else:
                    pix_out_mode = 0
                    pix_acc_mode = plan.acc_mode
                    pix_store_mode = plan.store_mode
                    pix_transfer_num = 1
                    # Phase 29: SD-UNet ky_outer>1 layers (and some decoder
                    # layers) override DS stride explicitly. Default to tile_h
                    # for FSRCNN / non-overridden Template A/B compatibility.
                    pix_stride = (
                        plan.ds_stride if plan.ds_stride is not None
                        else plan.tile_h
                    )
                    pix_bicubic_add = 0
                # Phase 31: optional ds_transfer_num override (applies to
                # standard conv DS path; not used by mask/pixshuffle since
                # those branches set transfer_num explicitly).
                if not plan.is_mask and not plan.is_pixelshuffle and not is_pixshuffle_legacy and plan.ds_transfer_num is not None:
                    pix_transfer_num = plan.ds_transfer_num
                # Pooling DS field selection. is_pool_store (deformable) takes
                # precedence over is_pool_out (pre-pool2d) — the two cases are
                # mutually exclusive in the supported models, but if both were
                # ever set, deformable's mode=3 is the correct value.
                is_pooling_val = 1 if (is_pool_store or is_pool_out or plan.is_mask) else 0
                pom = (4 if plan.is_mask else
                       (3 if is_pool_store else
                        (plan.pool_output_mode if is_pool_out else 0)))
                # pooling_out_new: for pool_output_mode >= 1, alternates 1/0
                # per DS step (load_idx parity). Mode 0 always uses 0.
                if is_pool_out and plan.pool_output_mode >= 1:
                    pooling_out_new = 1 if (load_idx % 2 == 0) else 0
                else:
                    pooling_out_new = 0
                # Phase 22/23: base_addr_pooling. For pool-preceding convs, the
                # DataStorer must also write the pooled result into a separate
                # SRAM region. The address breakdown:
                #   pool_addr_start                  — base for first DS of layer
                #   macro_idx * pool_addr_macro_stride — per-macro-W-tile offset
                #     (idx=2 conv1_2 right half: +2 over left half — golden line 564
                #      "datastorermanager.base_addr_pooling_cur = 144*4*2 + 2")
                #   (load_idx // pool_addr_inc_period) * pool_addr_stride
                #                                   — per-DS-step (or DS-pair-step)
                #     increment. Some layers (h_out_per_step=2, e.g. idx=2) update
                #     every DS (period=1), others (h_out_per_step=1, e.g. idx=5/8/11)
                #     update every 2 DS (period=2) because the hardware accumulates
                #     two conv-output rows into one pooled row.
                # For non-pool-output layers, this stays 0 (deformable conv overrides
                # to plan.tile_h * 2 in its dedicated emit path).
                # For mask-store (conv18 → unet_output_reg): the incrementing
                # address goes into base_addr_pooling (not base_addrs_res).
                # Golden: base_addr_pooling increments, base_addrs_res=0.
                if plan.is_mask:
                    base_addr_pooling = st.storer_bas_addr
                elif plan.has_pool_output:
                    base_addr_pooling = (
                        plan.pool_addr_start
                        + macro_idx * plan.pool_addr_macro_stride
                        + (load_idx // max(1, plan.pool_addr_inc_period)) * plan.pool_addr_stride
                    )
                else:
                    base_addr_pooling = 0
                # Phase 28 Fix 1: mask-store fields. is_new alternates per
                # cal_idx % storer_increment_period (golden line 2270/2460:
                # "is_new = 1 if cal_idx % 4 == 0 else 0").
                is_mask_field = 1 if plan.is_mask else 0
                is_new_field = (
                    1 if (plan.is_mask and (load_idx % plan.storer_increment_period == 0))
                    else 0
                )
                isa.DataStorer.dispatch(
                    quant_config_idx=st.quant_config_idx,
                    pixelshuffle_out_mode=pix_out_mode,
                    is_pixelshuffle=1 if is_pixshuffle else 0,
                    pooling_out_mode=pom,
                    pooling_out_new=pooling_out_new,
                    is_pooling=is_pooling_val,
                    reg_out_idx=st.acc_reg_idx,
                    acc_mode=pix_acc_mode,
                    transfer_num=pix_transfer_num,
                    store_mode=pix_store_mode,
                    stride=pix_stride,
                    base_addr_pooling=base_addr_pooling,
                    base_addrs_res=0 if plan.is_mask else st.storer_bas_addr,
                    is_bicubic_add=pix_bicubic_add,
                    is_first_or_last_row=0,
                    is_mask=is_mask_field,
                    is_new=is_new_field,
                    dest_buffer_idx=dest_buf,
                )
                st.acc_reg_idx = 1 - st.acc_reg_idx
                # Per-iteration DS base_addrs_res increment.
                # For mask-store layers (idx=22): only increment when
                # cal_idx % storer_increment_period == period-1 (golden line
                # 2274/2464: "if cal_idx % 4 == 3: base_addrs_res_cur += 2").
                # For all other layers: increment every cal_idx (period=1).
                if plan.is_mask:
                    if (load_idx % plan.storer_increment_period
                            == plan.storer_increment_period - 1):
                        st.storer_bas_addr += plan.storer_step
                else:
                    st.storer_bas_addr += plan.storer_step

            # Phase 20 P1: per-oc quant_config_idx toggle. The QL emitted
            # at the start of THIS oc iteration and the DS instructions just
            # emitted both used the pre-toggle value; toggling here means the
            # next oc iteration's QL/DS pair uses the alternated value.
            if plan.ql_per_oc_iter:
                st.quant_config_idx = 1 - st.quant_config_idx

    def _emit_group_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """Group conv emission: wraps the macro-tile body in level1 × level2 loops.

        Implements the four golden patterns in sd_sr_codegen.py:

          conv6  (g=2, single group_idx):
                level1=1, level2=2, QL inside level2 (one QL per group).
                DL bas_addr += group_idx * 2; DS base_addrs_res += group_idx * 18*8.

          conv7/8 (g=8, level1 only):
                level1=2, level2=1, QL inside level1 (one QL per outer block).
                DL bas_addr += level1_idx * dl_stride; DS += level1_idx * ds_stride.

          conv10 (g=8, true double loop):
                level1=2, level2=4, QL inside level2 (one QL per inner step).
                DL bas_addr += level1*9*4 + level2*1; DS += level1*144 + level2*36.

        Address overrides per (l1, l2) iteration are computed from
        plan.dl_level{1,2}_stride and plan.ds_level{1,2}_stride and passed to
        _emit_group_w_tile, which is otherwise structurally identical to
        _emit_w_macro_tile (single-shot, no group bookkeeping inside).
        """
        st = self.state
        # Destination ping-pong: terminal conv writes to the exported output
        # buffer; otherwise alternate 'a'/'b'. Same rule as _emit_standard_conv.
        if layer.idx == st.last_layer_idx:
            dest_buf = st.last_layer_dest_buffer
        else:
            dest_buf = "b" if st.feature_buf == "a" else "a"

        # Phase 28 Fix 4: per-group absolute DL base address.
        # When a group conv reads from MULTIPLE non-contiguous skip sources
        # (one source per group), each group_level2 iteration uses a DIFFERENT
        # absolute base — not `low_addr + l2 * stride`. This is the SD-UNet
        # idx=15 (golden conv11 g=2) pattern where group=0 reads from c10_out
        # at addr_map[14]=2016 and group=1 reads from c7_for_cat at
        # addr_map[11]=2160 (golden lines 1234, 1527).
        # Detection: layer has skip_sources whose count equals group_level2,
        # and addr_map has distinct addresses for those sources. The skip
        # source ordering in `layer.skip_sources` matches the IR concat input
        # order — the FIRST source corresponds to group_l2_idx=0 (golden's
        # group=0 reads from skip_sources[0]). For idx=15 the IR list is
        # [14, 11] with addrs [2016, 2160], so group 0 → 2016, group 1 → 2160.
        skip_srcs = getattr(layer, "skip_sources", None) or []
        per_group_dl_bases: Optional[List[int]] = None
        if (
            len(skip_srcs) == plan.group_level2
            and plan.group_level1 == 1
            and plan.group_level2 > 1
        ):
            # Resolve each skip source's address from the allocator map.
            # Same-address fallback (all 0) leaves per_group_dl_bases=None
            # so the standard `low + l2*stride` path runs.
            candidate = [self._addr_map.get(s, 0) for s in skip_srcs]
            if len(set(candidate)) > 1:
                per_group_dl_bases = candidate

        for l1 in range(plan.group_level1):
            if not plan.group_ql_in_level2:
                # conv7/8 pattern: emit QL once at the start of each level1
                # iteration (golden line 1118-1126 / 1316-1323).
                self.emit_quant_loader(
                    self.state.ql_layer_idx,
                    transnum=plan.quant_transnum,
                    quant_mode=plan.quant_mode,
                )
            for l2 in range(plan.group_level2):
                if plan.group_ql_in_level2:
                    # conv6 / conv10 pattern: emit QL inside level2 loop
                    # (golden line 1021-1028 for conv6, 1416-1423 for conv10).
                    self.emit_quant_loader(
                        self.state.ql_layer_idx,
                        transnum=plan.quant_transnum,
                        quant_mode=plan.quant_mode,
                    )
                # Compute group-specific DL/DS base offsets for this (l1,l2).
                ds_offset = l1 * plan.ds_level1_stride + l2 * plan.ds_level2_stride
                if per_group_dl_bases is not None:
                    # Phase 28 Fix 4: per-group absolute DL base from the
                    # allocator (skip_sources[l2] → its addr_map entry).
                    dl_base_for_group = per_group_dl_bases[l2]
                else:
                    dl_offset = l1 * plan.dl_level1_stride + l2 * plan.dl_level2_stride
                    dl_base_for_group = st.layer_input_bas_addr + dl_offset
                # Iterate macro W-tiles (group conv layers are typically W=18 or
                # W=9, so single-tile, but we honor plan.w_macro_tiles for
                # generality).
                for macro_idx, (w0, w_sz, bas_hint) in enumerate(plan.w_macro_tiles):
                    self._emit_group_w_tile(
                        layer, plan, w0, w_sz, bas_hint, macro_idx, dest_buf,
                        dl_base_override=dl_base_for_group + bas_hint,
                        ds_base_override=st.layer_output_bas_addr + ds_offset,
                        group_l1_idx=l1,
                    )
                # Advance weight base for the next group iteration.
                # Golden: weightloadermanager.bas_addr_cur[slot] += ky_outer * weight_transnum_base
                # (conv6 line 1061: +=3*24; conv7 line 1189: +=6*12; conv10 line 1499: +=3*24)
                st.weight_bas_addr[plan.wl_slot] += plan.weight_transnum_base * plan.cin_group * plan.ky_outer

        # Post-layer bookkeeping (mirrors _emit_standard_conv tail).
        st.feature_buf = dest_buf
        st.quant_config_idx = 1 - st.quant_config_idx
        st.last_feature_layer_idx = layer.idx

    def _emit_group_w_tile(
        self,
        layer: LayerDesc,
        plan: TilingPlan,
        w0: int,
        w_sz: int,
        bas_hint: int,
        macro_idx: int,
        dest_buf: str,
        *,
        dl_base_override: int,
        ds_base_override: int,
        group_l1_idx: int = 0,
    ) -> None:
        """Group conv variant of _emit_w_macro_tile.

        Identical to _emit_w_macro_tile except the initial DataLoader bas_addr
        and DataStorer base_addrs_res are taken from caller-supplied overrides
        (which already incorporate the group offset). The inner ky × cin loop
        body and per-cal_idx address increments are byte-for-byte identical.
        """
        st = self.state
        load_total = plan.load_total_num
        padding_num = plan.padding_num
        # Group-aware initial base addresses (caller pre-added group offset
        # plus the macro-tile bas_hint and layer base).
        st.dataloader_bas_addr = dl_base_override
        tile_half_offset = 0 if macro_idx == 0 else plan.tile_h * 4
        st.storer_bas_addr = ds_base_override + tile_half_offset

        for load_idx in range(load_total):
            for ky_g in range(plan.ky_outer):
                if plan.ky_outer > 1:
                    _pad_first = 1 if plan.line_buffer_reshape == 3 else 2
                    _pad_last  = 5 if plan.line_buffer_reshape == 3 else 2
                    if load_idx == 0 and ky_g == 0:
                        is_padding_row = _pad_first
                    elif load_idx == load_total - 1 and ky_g == plan.ky_outer - 1:
                        is_padding_row = _pad_last
                    else:
                        is_padding_row = 0
                else:
                    if load_idx < padding_num:
                        is_padding_row = 1
                    elif load_idx > load_total - 1 - padding_num:
                        is_padding_row = 5
                    else:
                        is_padding_row = 0
                for cin_g in range(plan.cin_group):
                    # Group conv DL address: use sliding-window for ky_outer>1
                    # (reshape!=3) layers, accumulator for reshape=3 (conv11).
                    if (
                        plan.ky_outer > 1
                        and plan.line_buffer_reshape != 3
                        and plan.dl_advance_nopad is not None
                    ):
                        _row = max(0, load_idx + ky_g - 1)
                        _per = plan.dl_advance_nopad // max(1, plan.cin_group)
                        _dl_off = _row * plan.dl_advance_nopad + cin_g * _per
                    elif plan.line_buffer_reshape == 3 and plan.ky_outer > 1:
                        # Phase 31: conv11 (reshape=3, ky_outer=3, cin_group=16)
                        # golden formula (sd_sr_codegen.py line 1255):
                        #   bas_addr = bas_addr_cur + 18*ic + ky
                        #              - (1 if cal_idx<pad and ky in (1,2) else 0)
                        # The -1 correction accounts for the upstream buffer's
                        # padding row layout where the first padded cal_idx step
                        # reads ky=1,2 from one row earlier (the actual first row
                        # of valid data, not the padding row).
                        _ky_correction = (
                            1
                            if (load_idx < padding_num and ky_g in (1, 2))
                            else 0
                        )
                        _dl_off = plan.tile_h * cin_g + ky_g - _ky_correction
                    else:
                        _dl_off = plan.tile_h * cin_g
                    isa.DataLoader.dispatch(
                        # Phase 29: golden DL.layer_idx = contiguous 0-based
                        # conv index (not the post-fusion layer.idx).
                        layer_idx=st.dl_layer_idx,
                        line_buffer_reshape=plan.line_buffer_reshape,
                        is_padding_row=is_padding_row,
                        read_mode=plan.read_mode,
                        transnum=(
                            plan.last_step_transnum
                            if plan.last_step_transnum is not None and load_idx == load_total - 1
                            else plan.line_buffer_rows
                        ),
                        line_buffer_idx=st.line_buffer_idx,
                        src_buffer_idx="offchip_input_buffer" if layer.idx == 0 else st.feature_buf,
                        bas_addr=st.dataloader_bas_addr + _dl_off,
                    )
                    isa.WeightLoader.dispatch(
                        acc_reg_comp_idx=st.acc_reg_idx,
                        kernal_size=0 if layer.k_h == 3 else 1,
                        line_buffer_row_shift=plan.wl_line_buffer_row_shift,
                        line_buffer_idx=st.line_buffer_idx,
                        is_padding_col=plan.wl_is_padding_col,
                        weight_parall_mode=plan.weight_parall_mode,
                        is_new=0 if (ky_g == 0 and cin_g == 0) else 1,
                        transnum=plan.weight_transnum_base,
                        bas_addr=st.weight_bas_addr[plan.wl_slot] + (ky_g * plan.cin_group + cin_g) * plan.weight_transnum_base,
                        is_bilinear_bicubic=plan.use_bilinear_weights,
                        offset_reg_idx=st.offset_reg_idx,
                    )
                    st.line_buffer_idx = 1 - st.line_buffer_idx

            # Advance accumulator only for non-sliding-window group conv layers.
            if not (
                plan.ky_outer > 1
                and plan.line_buffer_reshape != 3
                and plan.dl_advance_nopad is not None
            ):
                if plan.dl_advance_pad is not None:
                    st.dataloader_bas_addr += plan.dl_advance_pad if load_idx < padding_num else plan.dl_advance_nopad
                else:
                    st.dataloader_bas_addr += 2 if load_idx < padding_num else 4

            # See _emit_w_macro_tile for full mode-selection commentary; the
            # logic below is byte-for-byte identical for the group conv path.
            is_pool_store = (plan.store_mode == 3)        # deformable conv pool
            is_pool_out = plan.has_pool_output            # conv preceding pool2d
            is_pixshuffle_legacy = (plan.acc_mode == 5)
            is_pixshuffle = plan.is_pixelshuffle or is_pixshuffle_legacy
            if plan.is_pixelshuffle:
                # SD-UNet pre-DepthToSpace conv: explicit per-layer field values.
                pix_out_mode = plan.pixelshuffle_out_mode
                pix_acc_mode = plan.pixelshuffle_acc_mode
                pix_store_mode = plan.pixelshuffle_store_mode
                pix_transfer_num = plan.pixelshuffle_transfer_num
                pix_stride = plan.pixelshuffle_stride
                pix_bicubic_add = 0
            elif is_pixshuffle_legacy:
                # FSRCNN last_part legacy path.
                pix_out_mode = 1
                pix_acc_mode = plan.acc_mode
                pix_store_mode = plan.store_mode
                pix_transfer_num = 0
                pix_stride = 0
                pix_bicubic_add = 1
            else:
                pix_out_mode = 0
                pix_acc_mode = plan.acc_mode
                pix_store_mode = plan.store_mode
                pix_transfer_num = 1
                pix_stride = (
                    plan.ds_stride if plan.ds_stride is not None else plan.tile_h
                )
                pix_bicubic_add = 0
            # Phase 31: optional ds_transfer_num override. Applies to
            # group-conv DS path; pinned for SD-UNet conv8 (golden L=9)
            # which uses transfer_num=0 despite not being pixelshuffle.
            if plan.ds_transfer_num is not None:
                pix_transfer_num = plan.ds_transfer_num
            # Phase 32: last-DS-of-group transfer_num override. Used for
            # conv11 (L11, g=2): golden emits transfer_num=0 only on the
            # last cal_idx (sd_sr_codegen line 1285/1578).
            if plan.ds_last_transfer_num is not None and load_idx == load_total - 1:
                pix_transfer_num = plan.ds_last_transfer_num
            is_pooling_val = 1 if (is_pool_store or is_pool_out) else 0
            pom = 3 if is_pool_store else (plan.pool_output_mode if is_pool_out else 0)
            if is_pool_out and plan.pool_output_mode >= 1:
                pooling_out_new = 1 if (load_idx % 2 == 0) else 0
            else:
                pooling_out_new = 0
            # Phase 22/23: base_addr_pooling for pool-preceding group convs.
            # SD-UNet idx=11 (golden conv7, g=8) is the only pool-preceding
            # group conv: each group_level1 iteration starts at
            # `pool_addr_start + l1 * pool_addr_group_stride` and increments
            # by `pool_addr_stride` every `pool_addr_inc_period` DS steps.
            base_addr_pooling = (
                plan.pool_addr_start
                + group_l1_idx * plan.pool_addr_group_stride
                + macro_idx * plan.pool_addr_macro_stride
                + (load_idx // max(1, plan.pool_addr_inc_period)) * plan.pool_addr_stride
                if plan.has_pool_output else 0
            )
            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=pix_out_mode,
                is_pixelshuffle=1 if is_pixshuffle else 0,
                pooling_out_mode=pom,
                pooling_out_new=pooling_out_new,
                is_pooling=is_pooling_val,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=pix_acc_mode,
                transfer_num=pix_transfer_num,
                store_mode=pix_store_mode,
                stride=pix_stride,
                base_addr_pooling=base_addr_pooling,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=pix_bicubic_add,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx=dest_buf,
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += plan.storer_step

    def _emit_offset_gen(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """OffsetGenerator: QuantLoader + 3×(DataLoader+WeightLoader) + DataStorer(offset_reg).

        Matches sd_sr_codegen.py 'offset生成' sections.
        Reads pool output from buffer b at plan.data_bas_addr (fixed=64).
        Writes fused 18-channel offset map to dest_buffer_idx='offset_reg'.
        Uses weight slot [1] (separate from standard/deformable conv slot [0]).
        acc_reg_idx is NOT toggled inside the ky loop — only after DataStorer.
        """
        st = self.state
        self.emit_quant_loader(self.state.ql_layer_idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        st.dataloader_bas_addr = plan.data_bas_addr
        st.storer_bas_addr = 0

        for ky in range(plan.ky_outer):
            if ky == 0:
                is_padding_row = 3      # top-left corner (both H and W padding)
            elif ky == plan.ky_outer - 1:
                is_padding_row = 7      # bottom-right corner
            else:
                is_padding_row = 0

            isa.DataLoader.dispatch(
                # Phase 29: golden DL.layer_idx = contiguous 0-based conv index.
                layer_idx=st.dl_layer_idx,
                line_buffer_reshape=plan.line_buffer_reshape,
                is_padding_row=is_padding_row,
                read_mode=plan.read_mode,
                transnum=plan.line_buffer_rows,
                line_buffer_idx=st.line_buffer_idx,
                # offset_gen reads the upstream feature map currently in the
                # ping-pong buffer — follow the live feature_buf state.
                src_buffer_idx=st.feature_buf,
                bas_addr=st.dataloader_bas_addr,
            )
            isa.WeightLoader.dispatch(
                acc_reg_comp_idx=st.acc_reg_idx,
                kernal_size=0,
                line_buffer_row_shift=3,
                line_buffer_idx=st.line_buffer_idx,   # same as DataLoader (shared toggle)
                is_padding_col=5,
                weight_parall_mode=plan.weight_parall_mode,
                is_new=0 if ky == 0 else 1,
                transnum=plan.weight_transnum_base,
                bas_addr=st.weight_bas_addr[1] + ky * plan.weight_transnum_base,
                is_bilinear_bicubic=0,
                offset_reg_idx=0,
            )
            st.line_buffer_idx = 1 - st.line_buffer_idx

        isa.DataStorer.dispatch(
            quant_config_idx=st.quant_config_idx,
            pixelshuffle_out_mode=0,
            is_pixelshuffle=0,
            pooling_out_mode=0,
            pooling_out_new=0,
            is_pooling=0,
            reg_out_idx=st.acc_reg_idx,
            acc_mode=plan.acc_mode,
            transfer_num=1,
            store_mode=plan.store_mode,
            stride=0,
            base_addr_pooling=0,
            base_addrs_res=st.storer_bas_addr,
            is_bicubic_add=0,
            is_first_or_last_row=0,
            is_mask=0,
            is_new=0,
            dest_buffer_idx="offset_reg",
        )
        st.acc_reg_idx = 1 - st.acc_reg_idx
        st.quant_config_idx = 1 - st.quant_config_idx
        st.weight_bas_addr[1] += plan.weight_transnum_base * plan.ky_outer

    def _emit_deformable_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """
        Emission matching sd_sr_codegen.py mid_part deformable sequence.

        Structure per H-tile (cal_idx):
          for ky in range(ky_outer=3):
              OffsetLoader
              for ic_g in range(ic_inner=2):
                  DataLoader (6-row)
                  WeightLoader (bilinear)
          DataStorer (pooling mode)
        """
        st = self.state
        # Destination ping-pong: dconv reads from current feature_buf and writes
        # to the other. Terminal-layer override applies if a dconv is the final
        # compute layer (unusual, but supported for completeness).
        if layer.idx == st.last_layer_idx:
            dest_buf = st.last_layer_dest_buffer
        else:
            dest_buf = "b" if st.feature_buf == "a" else "a"

        self.emit_quant_loader(self.state.ql_layer_idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        cal_total = plan.load_total_num
        padding_num = plan.padding_num

        st.dataloader_bas_addr = st.layer_input_bas_addr
        st.storer_bas_addr = st.layer_output_bas_addr

        for cal_idx in range(cal_total):
            for ky in range(plan.ky_outer):
                isa.OffsetLoader.dispatch(
                    offset_reg_idx=st.offset_reg_idx,
                    bas_addr=cal_idx * plan.ky_outer + ky,
                )
                for ic_g in range(plan.ic_inner):
                    # Padding mode: transitions based on cal_idx and ky position
                    if cal_idx < padding_num and ky == 0:
                        is_padding_row = 4
                    elif cal_idx < padding_num and ky == 1:
                        is_padding_row = 1
                    elif cal_idx > cal_total - 1 - padding_num and ky == 1:
                        is_padding_row = 5
                    elif cal_idx > cal_total - 1 - padding_num and ky == 2:
                        is_padding_row = 6
                    else:
                        is_padding_row = 0

                    isa.DataLoader.dispatch(
                        # Phase 29: golden DL.layer_idx = contiguous 0-based
                        # conv index.
                        layer_idx=st.dl_layer_idx,
                        line_buffer_reshape=plan.line_buffer_reshape,
                        is_padding_row=is_padding_row,
                        read_mode=plan.read_mode,
                        transnum=plan.line_buffer_rows,
                        line_buffer_idx=st.line_buffer_idx,
                        # Deformable conv reads the live feature map — follow
                        # the ping-pong state rather than a hardcoded buffer.
                        src_buffer_idx=st.feature_buf,
                        bas_addr=st.dataloader_bas_addr + ic_g * plan.tile_h + (ky if cal_idx > 0 else 0),
                    )
                    # No toggle here — DataLoader and WeightLoader share the same line_buffer_idx.
                    # sr_codegen uses separate managers both starting at 0, so they stay in sync.

                    isa.WeightLoader.dispatch(
                        acc_reg_comp_idx=st.acc_reg_idx,
                        kernal_size=0,
                        line_buffer_row_shift=5,
                        line_buffer_idx=st.line_buffer_idx,  # same as DataLoader
                        is_padding_col=6,
                        weight_parall_mode=0,
                        is_new=0 if (ky == 0 and ic_g == 0) else 1,
                        transnum=plan.weight_transnum_base,
                        bas_addr=st.weight_bas_addr[plan.wl_slot] + plan.weight_transnum_base * (ky * plan.ic_inner + ic_g),
                        is_bilinear_bicubic=1,
                        offset_reg_idx=st.offset_reg_idx,
                    )
                    st.line_buffer_idx = 1 - st.line_buffer_idx  # single toggle after both

                st.offset_reg_idx = 1 - st.offset_reg_idx

            # Last deformable conv (acc_mode=2): no pooling, stride=0.
            # Other deformable convs (acc_mode=4): pool-while-store.
            is_last_dconv = (plan.acc_mode == 2)
            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=0,
                pooling_out_mode=0 if is_last_dconv else 3,
                pooling_out_new=0,
                is_pooling=0 if is_last_dconv else 1,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=plan.acc_mode,
                transfer_num=1,
                store_mode=plan.store_mode,
                stride=0 if is_last_dconv else plan.tile_h,
                base_addr_pooling=0 if is_last_dconv else plan.tile_h * 2,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx=dest_buf,
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += plan.storer_step
            # Phase 30: per-layer override; default keeps historic 2/4 (deformable path).
            if plan.dl_advance_pad is not None:
                st.dataloader_bas_addr += plan.dl_advance_pad if cal_idx < padding_num else plan.dl_advance_nopad
            else:
                st.dataloader_bas_addr += 2 if cal_idx < padding_num else 4

        st.weight_bas_addr[plan.wl_slot] += plan.weight_transnum_base * plan.ky_outer * plan.ic_inner
        # After all H-tiles write the output feature map, swap ping-pong state
        # so the next layer reads from dest_buf.
        st.feature_buf = dest_buf
        st.quant_config_idx = 1 - st.quant_config_idx
        st.last_feature_layer_idx = layer.idx

    def _emit_preamble(self) -> None:
        """5-instruction DDR preload preamble emitted once per model (is_first=True).

        Mirrors sd_inst(is_first=True): quant×2 (src_buffer_idx=2) + weight×3 (src_buffer_idx=1).
        These are instructions [0-4]; the image OffchipDataLoader becomes [5].
        """
        for load_model in range(2):
            isa.OffchipDataLoader.dispatch(
                transnum="unet_total",
                load_model=load_model,
                src_buffer_idx=2,
                bas_addr=0,
            )
        for load_model in range(3):
            isa.OffchipDataLoader.dispatch(
                transnum="unet_total",
                load_model=load_model,
                src_buffer_idx=1,
                bas_addr=0,
            )


def emit_program(
    layers: List[LayerDesc],
    plans: List[TilingPlan],
    *,
    addr_map: Optional[Dict[int, int]] = None,
    buf_map: Optional[Dict[int, str]] = None,
    is_first: bool = False,
    load_next: bool = False,
    emit_image_load: bool = True,
    emit_image_load_at_end: bool = False,
    image_transnum: int = 576,
    inter_layer_transnum: Optional[int] = None,
    inter_layer_bas_addr: int = 576,
    load_next_transnum: int = 64,
    load_next_load_model: int = 1,
    load_next_bas_addr: int = 576,
    emit_offchip_store: bool = False,
    offchip_store_src_buffer: str = "fsrcnn_output_buffer",
    offchip_store_transnum: int = 1024,
    offchip_store_base_addr: int = 0,
    last_layer_dest_buffer: str = "fsrcnn_output_buffer",
    initial_quant_bas_addr: int = 0,
    initial_weight_bas_addr: Optional[List[int]] = None,
    initial_layer0_input_bas_addr: int = 0,
    finalize: bool = True,
) -> List[Dict[str, Any]]:
    """Emit full network.

    Args:
        is_first: Emit 5-instruction DDR preamble (quant/weight preload) — only for
                  the very first frame of a multi-frame pipeline.
        load_next: After layer-0 tiles, emit an OffchipDataLoader for the NEXT frame's
                   image so the hardware can prefetch it while the current frame continues.
        emit_image_load: Emit an OffchipDataLoader before layer-0 tiles to DMA the input
                         image from DDR. Set False when the image is pre-loaded by a
                         preceding pipeline stage (e.g. FSRCNN golden sr_inst() scenario
                         where UNet already populated offchip_input_buffer).
        image_transnum: Pixel count for one image tile (default 144×4=576 for UNet 144-row).
                        Resolved by the pipeline.py wrapper from layers[0] when applicable.
        inter_layer_transnum: If not None, emit an OffchipDataLoader for the inter-model
                              input (e.g. FSRCNN input) after layer-0 tiles.
                              Set to 64 (32×2) for the UNet→FSRCNN boundary.
        inter_layer_bas_addr: bas_addr for the inter-layer OffchipDataLoader. Defaults to
                              image_transnum (the next-image is placed after the first
                              model's image in the on-chip feature buffer).
        emit_offchip_store: If True, emit a final OffchipDataStorer after all layers —
                            FSRCNN's sr_inst() ends with this instruction to write the
                            SR result from on-chip output buffer back to DDR. UNet's
                            sd_inst() does NOT emit one (its output stays on-chip as
                            input to the downstream FSRCNN/bicubic stage).
        offchip_store_src_buffer: Source buffer name for the terminal OffchipDataStorer.
                                  Golden FSRCNN uses 'fsrcnn_output_buffer'.
        offchip_store_transnum: Transfer count for the terminal OffchipDataStorer.
                                Golden FSRCNN uses 1024 (= 32×32 SR output block).
        offchip_store_base_addr: DDR base address for the terminal OffchipDataStorer.
        last_layer_dest_buffer: dest_buffer_idx for the DataStorer of the final
                                conv/deformable_conv layer. All preceding conv
                                layers ping-pong between 'a' and 'b'; the last
                                one writes to this named buffer so the terminal
                                OffchipDataStorer can drain it to DDR.
        initial_quant_bas_addr: Seed value for EmitterState.quant_bas_addr. Default 0.
                                Set to 665 for FSRCNN-only sr_inst() golden mode (the
                                upstream UNet has already consumed addresses 0..664
                                in the on-chip quant buffer, see sd_sr_codegen.py
                                line 2491: ``quantloadermanager.bas_addr_cur = 665``).
        initial_weight_bas_addr: Seed value for EmitterState.weight_bas_addr (list of
                                 3 ints, one per weight_parall_mode). Default [0, 0, 0].
                                 Set to [1737, 792, 1152] for FSRCNN-only sr_inst()
                                 golden mode (sd_sr_codegen.py line 2490:
                                 ``weightloadermanager.bas_addr_cur = [1737, 792, 1152]``).
        initial_layer0_input_bas_addr: Override layer-0's resolved layer_input_bas_addr.
                                       Default 0. Set to image_transnum (576) for
                                       FSRCNN-only mode where the FSRCNN input image
                                       lives at offset image_transnum in the on-chip
                                       feature buffer (after the upstream UNet image).
        finalize: Run dependency-analysis + virtual-register-allocation post-pass.
    """
    em = InstructionEmitter(addr_map=addr_map, buf_map=buf_map)
    em.reset()
    em._addr_map = addr_map or {}
    em._buf_map = buf_map or {}

    # FSRCNN-only seed: pre-load EmitterState's running-base counters so the
    # first QL/WL emission lands on the golden sr_inst() initial values
    # (UNet has already consumed the lower portions of these on-chip buffers).
    if initial_quant_bas_addr:
        em.state.quant_bas_addr = initial_quant_bas_addr
    if initial_weight_bas_addr is not None:
        em.state.weight_bas_addr = list(initial_weight_bas_addr)
    # Layer-0 input override: emit_layer resolves layer_input_bas_addr from
    # addr_map[last_feature_layer_idx]. With last_feature_layer_idx=-1 (initial)
    # and an empty addr_map entry for -1, the lookup returns 0. Inject the
    # override here so layer 0's DataLoader bas_addr starts at image_transnum
    # (576) for FSRCNN-only, matching golden sr_inst().
    if initial_layer0_input_bas_addr:
        em._addr_map = dict(em._addr_map)
        em._addr_map.setdefault(-1, initial_layer0_input_bas_addr)

    # Ping-pong buffer allocator: identify the terminal compute layer so its
    # DataStorer targets last_layer_dest_buffer instead of 'a'/'b'.
    conv_layers = [L for L in layers if L.op in ("conv2d", "deformable_conv2d")]
    if conv_layers:
        em.state.last_layer_idx = conv_layers[-1].idx
        em.state.last_layer_dest_buffer = last_layer_dest_buffer

    if is_first:
        em._emit_preamble()

    for i, (L, P) in enumerate(zip(layers, plans)):
        if i == 0 and emit_image_load and not emit_image_load_at_end:
            isa.OffchipDataLoader.dispatch(
                transnum=image_transnum,
                load_model=0,
                src_buffer_idx=0,
                bas_addr=0,
            )

        em.emit_layer(L, P)

        if i == 0:
            # Phase 31: SD-UNet archived golden emits the image OffchipDataLoader
            # at the END of L=0 (after all layer-0 DSes), not the start. This
            # path matches the archived `pseudo_code_load_next_mid.txt` bucket
            # alignment exactly. Gated on `emit_image_load_at_end` so FSRCNN
            # (which emits at start, then runs no L=0 conv body until later)
            # is unaffected.
            if emit_image_load and emit_image_load_at_end:
                isa.OffchipDataLoader.dispatch(
                    transnum=image_transnum,
                    load_model=0,
                    src_buffer_idx=0,
                    bas_addr=0,
                )
            if load_next:
                isa.OffchipDataLoader.dispatch(
                    transnum=load_next_transnum,
                    load_model=load_next_load_model,
                    src_buffer_idx=0,
                    bas_addr=load_next_bas_addr,
                )
            if inter_layer_transnum is not None:
                isa.OffchipDataLoader.dispatch(
                    transnum=inter_layer_transnum,
                    load_model=1,
                    src_buffer_idx=0,
                    bas_addr=inter_layer_bas_addr,
                )

    # FSRCNN terminal off-chip write-back — matches sr_inst()'s final
    # OffchipDataStorer at sd_sr_codegen.py line ~3672. The post-pass
    # dependency analyzer already wires this to the last DataStorer whose
    # dest_buffer_idx == offchip_store_src_buffer (see post_pass.py:209).
    if emit_offchip_store:
        isa.OffchipDataStorer.dispatch(
            src_buffer=offchip_store_src_buffer,
            transnum=offchip_store_transnum,
            base_addr=offchip_store_base_addr,
        )

    raw: List[Dict[str, Any]] = list(isa.Inst.code_list)
    if finalize:
        finalize_instructions(raw)
    return raw
