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


class InstructionEmitter:
    """Dispatch emit_layer to per-op template methods."""

    def __init__(self, state: Optional[EmitterState] = None):
        self.state = state or EmitterState()

    def reset(self) -> None:
        isa.reset_instruction_stream()
        self.state = EmitterState()

    def emit_layer(self, layer: LayerDesc, plan: TilingPlan) -> None:
        if layer.op in ("conv2d", "deformable_conv2d", "offset_gen"):
            self.state.conv_layer_counter += 1
            if layer.op == "deformable_conv2d":
                self._emit_deformable_conv(layer, plan)
            elif layer.op == "offset_gen":
                self._emit_offset_gen(layer, plan)
            else:
                self._emit_standard_conv(layer, plan)
        elif layer.op == "pool2d":
            isa.Inst.code_list.append({
                "code_num": [isa.Inst.current_code_num],
                "op_code": "PseudoOp",
                "note": "skipped-pool2d",
                "layer_idx": layer.idx,
            })
            isa.Inst.current_code_num += 1

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
        st = self.state
        # Destination ping-pong buffer: terminal conv writes to the exported
        # output buffer; all other layers alternate 'a'/'b'.
        if layer.idx == st.last_layer_idx:
            dest_buf = st.last_layer_dest_buffer
        else:
            dest_buf = "b" if st.feature_buf == "a" else "a"

        self.emit_quant_loader(layer.idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        for macro_idx, (w0, w_sz, bas_hint) in enumerate(plan.w_macro_tiles):
            self._emit_w_macro_tile(layer, plan, w0, w_sz, bas_hint, macro_idx, dest_buf)
        st.weight_bas_addr[0] += plan.weight_transnum_base * plan.cin_group * plan.ky_outer
        # Toggle feature_buf so the next layer reads from the buffer we just
        # wrote to. (Safe even after the terminal layer — nothing reads it.)
        st.feature_buf = dest_buf
        # QuantLoader and all DataStorers in this layer share the same
        # quant_config_idx; toggle once here after all tiles are done.
        st.quant_config_idx = 1 - st.quant_config_idx

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
        st.dataloader_bas_addr = bas_hint
        # Right-half output base: matches sd_codegen h_in*4 offset
        st.storer_bas_addr = 0 if macro_idx == 0 else plan.tile_h * 4

        for load_idx in range(load_total):
            # ky × cin inner loops.
            # plan.ky_outer == 1 for most standard convs (hardware handles 3×3
            # kernel implicitly via the line buffer).  plan.ky_outer == 3 for
            # Template F (large-cin 3×3 with small cout, e.g. FSRCNN L11) where
            # the kernel-row dimension is explicit in software.
            # For multi-ky templates, padding applies per (outer_tile, ky_g):
            #   first outer tile AND ky=0 → is_padding_row=1
            #   last outer tile AND ky=ky_outer-1 → is_padding_row=5
            # For single-ky templates, padding is per outer tile (all cin_g share).
            for ky_g in range(plan.ky_outer):
                if plan.ky_outer > 1:
                    if load_idx == 0 and ky_g == 0:
                        is_padding_row = 1
                    elif load_idx == load_total - 1 and ky_g == plan.ky_outer - 1:
                        is_padding_row = 5
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
                    isa.DataLoader.dispatch(
                        layer_idx=layer.idx,
                        line_buffer_reshape=plan.line_buffer_reshape,
                        is_padding_row=is_padding_row,
                        read_mode=plan.read_mode,
                        transnum=plan.line_buffer_rows,
                        line_buffer_idx=st.line_buffer_idx,
                        # Layer 0 sources from the DDR-preloaded offchip input
                        # buffer; subsequent layers follow ping-pong alternation.
                        src_buffer_idx="offchip_input_buffer" if layer.idx == 0 else st.feature_buf,
                        bas_addr=st.dataloader_bas_addr + plan.tile_h * cin_g,
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
                        bas_addr=st.weight_bas_addr[0] + (ky_g * plan.cin_group + cin_g) * plan.weight_transnum_base,
                        is_bilinear_bicubic=plan.use_bilinear_weights,
                        offset_reg_idx=st.offset_reg_idx,
                    )
                    st.line_buffer_idx = 1 - st.line_buffer_idx

            # Advance DataLoader base ONCE per cal_idx (outer H step), not per cin_g.
            st.dataloader_bas_addr += 2 if load_idx < padding_num else 4

            # DataStorer: mode-specific fields derived from store_mode / acc_mode.
            #   store_mode=3 (pool-while-store): is_pooling=1, pooling_out_mode=3
            #   acc_mode=5  (pixelshuffle output): is_pixelshuffle=1, stride=0, transfer_num=0, is_bicubic_add=1, pixelshuffle_out_mode=1
            is_pool_store = (plan.store_mode == 3)
            is_pixshuffle = (plan.acc_mode == 5)
            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=1 if is_pixshuffle else 0,
                is_pixelshuffle=1 if is_pixshuffle else 0,
                pooling_out_mode=3 if is_pool_store else 0,
                pooling_out_new=0,
                is_pooling=1 if is_pool_store else 0,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=plan.acc_mode,
                transfer_num=0 if is_pixshuffle else 1,
                store_mode=plan.store_mode,
                stride=0 if is_pixshuffle else plan.tile_h,
                base_addr_pooling=0,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=1 if is_pixshuffle else 0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx=dest_buf,
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            # Per-iteration DS base_addrs_res increment. Was hardcoded as 2
            # (Template A/B default) — now driven by plan.storer_step so each
            # template uses its golden-correct stride (C/D=1, E=4, A/B=2,
            # pixelshuffle=128).
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
        self.emit_quant_loader(layer.idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
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
                layer_idx=layer.idx,
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

        self.emit_quant_loader(layer.idx, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        cal_total = plan.load_total_num
        padding_num = plan.padding_num

        st.dataloader_bas_addr = 0
        st.storer_bas_addr = 0

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
                        layer_idx=layer.idx,
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
                        bas_addr=st.weight_bas_addr[0] + plan.weight_transnum_base * (ky * plan.ic_inner + ic_g),
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
            st.dataloader_bas_addr += 2 if cal_idx < padding_num else 4

        st.weight_bas_addr[0] += plan.weight_transnum_base * plan.ky_outer * plan.ic_inner
        # After all H-tiles write the output feature map, swap ping-pong state
        # so the next layer reads from dest_buf.
        st.feature_buf = dest_buf
        st.quant_config_idx = 1 - st.quant_config_idx

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
    is_first: bool = False,
    load_next: bool = False,
    emit_image_load: bool = True,
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
        inter_layer_transnum: If not None, emit an OffchipDataLoader for the inter-model
                              input (e.g. FSRCNN input) after layer-0 tiles.
                              Set to 64 (32×2) for the UNet→FSRCNN boundary.
        inter_layer_bas_addr: bas_addr for the inter-layer OffchipDataLoader.
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
        finalize: Run dependency-analysis + virtual-register-allocation post-pass.
    """
    em = InstructionEmitter()
    em.reset()

    # Ping-pong buffer allocator: identify the terminal compute layer so its
    # DataStorer targets last_layer_dest_buffer instead of 'a'/'b'.
    conv_layers = [L for L in layers if L.op in ("conv2d", "deformable_conv2d")]
    if conv_layers:
        em.state.last_layer_idx = conv_layers[-1].idx
        em.state.last_layer_dest_buffer = last_layer_dest_buffer

    if is_first:
        em._emit_preamble()

    for i, (L, P) in enumerate(zip(layers, plans)):
        if i == 0 and emit_image_load:
            isa.OffchipDataLoader.dispatch(
                transnum=image_transnum,
                load_model=0,
                src_buffer_idx=0,
                bas_addr=0,
            )

        em.emit_layer(L, P)

        if i == 0:
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
