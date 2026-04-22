"""Tiling decisions derived from docs/unet_fsrcnn_tiling_and_codegen_guide.md.

Applied as a separate stage between LayerDesc extraction and ISA emission.
The TilingPlan carries all parameters needed by the emitter — no magic numbers
should appear in emitter.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ir.layer_desc import LayerDesc


def _pick_w_micro_tile(w_in: int) -> int:
    """Micro-tile width inside a macro tile: 32 / 64 / 128."""
    if w_in >= 128:
        return 128
    if w_in >= 64:
        return 64
    return 32


def _macro_w_tiles(w_in: int) -> List[Tuple[int, int, int]]:
    """
    Return (w_start, w_size, bas_addr_hint) for each horizontal macro tile.

    Guide §5.1: W=256 → two 128-wide halves.
    bas_addr_hint for right half uses +288 step (matches sd_codegen).
    """
    if w_in <= 128:
        return [(0, w_in, 0)]
    chunks = []
    start = 0
    hint = 0
    while start < w_in:
        sz = min(128, w_in - start)
        chunks.append((start, sz, hint))
        start += sz
        hint += 288 if sz == 128 else sz * 2
    return chunks


@dataclass
class TilingPlan:
    """Per-layer tiling + ISA template parameters (guide-aligned)."""

    layer_idx: int
    h_out_per_step: int          # output rows advanced per outer H step
    load_total_num: int          # DataLoader blocks along H per macro W tile
    padding_num: int             # first/last N blocks need padding flags
    line_buffer_rows: int        # rows loaded per DataLoader block (4=std, 6=deformable)
    line_buffer_reshape: int     # 0-3 line buffer reshape mode
    w_macro_tiles: List[Tuple[int, int, int]]
    w_micro_tile: int            # 32 / 64 / 128
    cin_group: int               # input channel group size (2/4/8)
    cout_group: int              # output channel group size (8/16/32)
    weight_parall_mode: int      # 0/1 MAC array upper/lower half select
    weight_transnum_base: int    # WeightLoader transnum for one cin group
    read_mode: int               # DataLoader read_mode field
    use_bilinear_weights: int    # WeightLoader.is_bilinear_bicubic (0/1)
    ky_outer: int                # deformable: ky loop count (3 for 3×3 kernel)
    ic_inner: int                # deformable: ic groups per ky step
    # DataStorer / QuantLoader params (must be set correctly for golden parity)
    acc_mode: int = 0            # 0=standard, 4=deformable pooling
    store_mode: int = 0          # 0=standard, 3=deformable
    quant_mode: int = 0          # QuantLoader.quant_mode
    quant_transnum: int = 4      # QuantLoader.transnum
    data_bas_addr: int = 0       # DataLoader base address (offset_gen: fixed at 64)
    notes: str = ""


def choose_tiling(layer: LayerDesc) -> TilingPlan:
    """Map LayerDesc → TilingPlan using documented UNet/FSRCNN templates."""

    # OffsetGenerator: fused pool2d+conv2d that writes to offset_reg.
    # All parameters are fixed by the FSRCNN 8→18 offset-conv spec (golden sd_sr_codegen.py).
    # data_bas_addr=64 (=32×2): fixed address in buffer b where pool output is stored.
    if layer.op == "offset_gen":
        return TilingPlan(
            layer_idx=layer.idx,
            h_out_per_step=1,
            load_total_num=3,
            padding_num=1,
            line_buffer_rows=4,
            line_buffer_reshape=2,
            w_macro_tiles=[(0, layer.w_in, 0)],
            w_micro_tile=_pick_w_micro_tile(layer.w_in),
            cin_group=4,
            cout_group=8,
            weight_parall_mode=1,
            weight_transnum_base=24,
            read_mode=1,
            use_bilinear_weights=0,
            ky_outer=3,
            ic_inner=1,
            acc_mode=1,
            store_mode=1,
            quant_mode=2,
            quant_transnum=16,
            data_bas_addr=64,
            notes="offset_gen — OffsetGenerator subgraph (pool2d+conv2d fused)",
        )

    # Non-conv layers: degenerate plans
    if layer.op in ("relu", "prelu"):
        return TilingPlan(
            layer_idx=layer.idx,
            h_out_per_step=1,
            load_total_num=1,
            padding_num=0,
            line_buffer_rows=1,
            line_buffer_reshape=0,
            w_macro_tiles=[(0, layer.w_in, 0)],
            w_micro_tile=32,
            cin_group=1,
            cout_group=1,
            weight_parall_mode=0,
            weight_transnum_base=1,
            read_mode=0,
            use_bilinear_weights=0,
            ky_outer=1,
            ic_inner=1,
            notes="activation — no conv tiling",
        )

    if layer.op == "pool2d":
        rows = max(1, layer.h_in // 4)
        return TilingPlan(
            layer_idx=layer.idx,
            h_out_per_step=4,
            load_total_num=rows,
            padding_num=1,
            line_buffer_rows=4,
            line_buffer_reshape=0,
            w_macro_tiles=_macro_w_tiles(layer.w_in),
            w_micro_tile=_pick_w_micro_tile(layer.w_in),
            cin_group=1,
            cout_group=8,
            weight_parall_mode=0,
            weight_transnum_base=1,
            read_mode=0,
            use_bilinear_weights=0,
            ky_outer=1,
            ic_inner=1,
            notes="pool2d",
        )

    h_in, w_in = layer.h_in, layer.w_in
    cin, cout = layer.cin, layer.cout
    k = layer.k_h

    # Guide §3.1: deformable / bilinear uses 6-row line buffer; standard conv uses 4.
    if layer.deformable:
        line_rows = 6
        read_mode = 0
        bilinear = 1
        ky_outer = 3 if k == 3 else 1
        ic_inner = 2 if cin > 1 else 1
        h_out_per_step = 4
        load_total_num = max(1, h_in // h_out_per_step)
        padding_num = 1
        acc_mode = 4        # pooling accumulate mode for deformable
        store_mode = 3      # deformable store mode
        # line_buffer_reshape=1 for sub-128 widths (guide §3.2)
        lbr = 1 if w_in <= 128 else 0
    else:
        line_rows = 4
        read_mode = 0
        bilinear = 0
        ky_outer = 1
        ic_inner = 1
        # Template A/B: two output rows per outer step → H/2 loader iterations
        h_out_per_step = 2
        load_total_num = max(1, h_in // h_out_per_step)
        padding_num = 1
        acc_mode = 0
        store_mode = 0
        # line_buffer_reshape=1 for 128×72 and smaller (guide §3.2)
        lbr = 1 if w_in <= 128 else 0

    # Channel grouping (guide §3.3)
    if cin <= 1:
        cin_group = 1
    elif cin <= 4:
        cin_group = 4
    elif cin <= 8:
        cin_group = 4
    else:
        cin_group = 8

    if cout <= 4:
        cout_g = 4
    elif cout <= 8:
        cout_g = 8
    else:
        cout_g = min(32, max(8, (cout + 7) // 8 * 8))

    w_micro = _pick_w_micro_tile(w_in)
    macros = _macro_w_tiles(w_in)

    # WeightLoader transnum: 9 per cin group for 3×3; exactly 12 for deformable bilinear
    if layer.deformable:
        wt = 12
    elif k == 3 and cin == 1:
        wt = 9
    elif k == 3:
        wt = 9
    elif k == 1:
        wt = 1
    else:
        wt = k * k

    wpar = 0 if cout <= 8 else 1

    return TilingPlan(
        layer_idx=layer.idx,
        h_out_per_step=h_out_per_step,
        load_total_num=load_total_num,
        padding_num=padding_num,
        line_buffer_rows=line_rows,
        line_buffer_reshape=lbr,
        w_macro_tiles=macros,
        w_micro_tile=w_micro,
        cin_group=cin_group,
        cout_group=cout_g,
        weight_parall_mode=wpar,
        weight_transnum_base=wt,
        read_mode=read_mode,
        use_bilinear_weights=bilinear,
        ky_outer=ky_outer,
        ic_inner=ic_inner,
        acc_mode=acc_mode,
        store_mode=store_mode,
        quant_mode=0,
        quant_transnum=4,
        notes="deformable" if layer.deformable else "standard conv",
    )


def plan_all(layers: List[LayerDesc]) -> List[TilingPlan]:
    return [choose_tiling(L) for L in layers]
