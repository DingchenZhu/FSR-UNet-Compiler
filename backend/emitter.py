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
        elif layer.op in ("relu", "prelu", "pool2d"):
            isa.Inst.code_list.append({
                "code_num": [isa.Inst.current_code_num],
                "op_code": "PseudoOp",
                "note": f"skipped-{layer.op}",
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

    def _emit_standard_conv(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """Template A: QuantLoader → per macro W-tile (DataLoader, WeightLoader, DataStorer)."""
        self.emit_quant_loader(self.state.conv_layer_counter, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
        for macro_idx, (w0, w_sz, bas_hint) in enumerate(plan.w_macro_tiles):
            self._emit_w_macro_tile(layer, plan, w0, w_sz, bas_hint, macro_idx)

    def _emit_w_macro_tile(
        self,
        layer: LayerDesc,
        plan: TilingPlan,
        w0: int,
        w_sz: int,
        bas_hint: int,
        macro_idx: int,
    ) -> None:
        st = self.state
        load_total = plan.load_total_num
        padding_num = plan.padding_num
        st.dataloader_bas_addr = bas_hint
        # Right-half output base: matches sd_codegen h_in*4 offset
        st.storer_bas_addr = 0 if macro_idx == 0 else layer.h_in * 4

        for load_idx in range(load_total):
            if load_idx < padding_num:
                is_padding_row = 1
            elif load_idx > load_total - 1 - padding_num:
                is_padding_row = 5
            else:
                is_padding_row = 0

            # DataLoader and WeightLoader use the SAME line_buffer_idx per iteration.
            # sd_codegen uses separate managers both starting at 0 and toggling independently,
            # so they always stay in sync. We replicate this with a single toggle AFTER both.
            isa.DataLoader.dispatch(
                layer_idx=layer.idx,
                line_buffer_reshape=plan.line_buffer_reshape,
                is_padding_row=is_padding_row,
                read_mode=plan.read_mode,
                transnum=plan.line_buffer_rows,
                line_buffer_idx=st.line_buffer_idx,
                src_buffer_idx="offchip_input_buffer" if layer.idx == 0 else "a",
                bas_addr=st.dataloader_bas_addr,
            )
            st.dataloader_bas_addr += 2 if load_idx < padding_num else 4

            isa.WeightLoader.dispatch(
                acc_reg_comp_idx=st.acc_reg_idx,
                kernal_size=0 if layer.k_h == 3 else 1,
                line_buffer_row_shift=1,
                line_buffer_idx=st.line_buffer_idx,   # same as DataLoader (no toggle between)
                is_padding_col=1,
                weight_parall_mode=plan.weight_parall_mode,
                is_new=1,
                transnum=plan.weight_transnum_base,
                bas_addr=st.weight_bas_addr[0],
                is_bilinear_bicubic=plan.use_bilinear_weights,
                offset_reg_idx=st.offset_reg_idx,
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.line_buffer_idx = 1 - st.line_buffer_idx  # single toggle after both

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
                stride=layer.h_in,
                base_addr_pooling=0,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx="a",
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += 2

    def _emit_offset_gen(self, layer: LayerDesc, plan: TilingPlan) -> None:
        """OffsetGenerator: QuantLoader + 3×(DataLoader+WeightLoader) + DataStorer(offset_reg).

        Matches sd_sr_codegen.py 'offset生成' sections.
        Reads pool output from buffer b at plan.data_bas_addr (fixed=64).
        Writes fused 18-channel offset map to dest_buffer_idx='offset_reg'.
        Uses weight slot [1] (separate from standard/deformable conv slot [0]).
        acc_reg_idx is NOT toggled inside the ky loop — only after DataStorer.
        """
        st = self.state
        self.emit_quant_loader(st.conv_layer_counter, transnum=plan.quant_transnum, quant_mode=plan.quant_mode)
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
                src_buffer_idx="b",
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
        self.emit_quant_loader(self.state.conv_layer_counter, transnum=max(plan.quant_transnum, layer.cout), quant_mode=plan.quant_mode)
        cal_total = max(1, layer.h_in // plan.h_out_per_step)
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
                        src_buffer_idx="b",
                        bas_addr=st.dataloader_bas_addr + ic_g * 32 + (ky if cal_idx > 0 else 0),
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

            isa.DataStorer.dispatch(
                quant_config_idx=st.quant_config_idx,
                pixelshuffle_out_mode=0,
                is_pixelshuffle=0,
                pooling_out_mode=3,
                pooling_out_new=0,
                is_pooling=1,
                reg_out_idx=st.acc_reg_idx,
                acc_mode=plan.acc_mode,
                transfer_num=1,
                store_mode=plan.store_mode,
                stride=32,
                base_addr_pooling=layer.h_in * 2,
                base_addrs_res=st.storer_bas_addr,
                is_bicubic_add=0,
                is_first_or_last_row=0,
                is_mask=0,
                is_new=0,
                dest_buffer_idx="a",
            )
            st.acc_reg_idx = 1 - st.acc_reg_idx
            st.storer_bas_addr += 4
            st.dataloader_bas_addr += 2 if cal_idx < padding_num else 4

        st.weight_bas_addr[0] += plan.weight_transnum_base * plan.ky_outer * plan.ic_inner

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
    image_transnum: int = 576,
    inter_layer_transnum: Optional[int] = None,
    inter_layer_bas_addr: int = 576,
    finalize: bool = True,
) -> List[Dict[str, Any]]:
    """Emit full network.

    Args:
        is_first: Emit 5-instruction DDR preamble (quant/weight preload) — only for
                  the very first frame of a multi-frame pipeline.
        load_next: After layer-0 tiles, emit an OffchipDataLoader for the NEXT frame's
                   image so the hardware can prefetch it while the current frame continues.
        image_transnum: Pixel count for one image tile (default 144×4=576 for UNet 144-row).
        inter_layer_transnum: If not None, emit an OffchipDataLoader for the inter-model
                              input (e.g. FSRCNN input) after layer-0 tiles.
                              Set to 64 (32×2) for the UNet→FSRCNN boundary.
        inter_layer_bas_addr: bas_addr for the inter-layer OffchipDataLoader.
        finalize: Run dependency-analysis + virtual-register-allocation post-pass.
    """
    em = InstructionEmitter()
    em.reset()

    if is_first:
        em._emit_preamble()

    for i, (L, P) in enumerate(zip(layers, plans)):
        if i == 0:
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
                    transnum=image_transnum,
                    load_model=0,
                    src_buffer_idx=0,
                    bas_addr=0,
                )
            if inter_layer_transnum is not None:
                isa.OffchipDataLoader.dispatch(
                    transnum=inter_layer_transnum,
                    load_model=1,
                    src_buffer_idx=0,
                    bas_addr=inter_layer_bas_addr,
                )

    raw: List[Dict[str, Any]] = list(isa.Inst.code_list)
    if finalize:
        finalize_instructions(raw)
    return raw
