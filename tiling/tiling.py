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
    line_buffer_rows: int        # rows loaded per DataLoader block (template-specific)
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
    tile_h: int = 32             # hardware spatial tile height (rows processed per burst)
    wl_line_buffer_row_shift: int = 1   # WeightLoader.line_buffer_row_shift
    wl_is_padding_col: int = 1          # WeightLoader.is_padding_col
    # DataStorer.base_addrs_res per-iteration increment (was hardcoded as +=2
    # in emitter._emit_w_macro_tile). Template-specific values from golden
    # sd_sr_codegen.py:
    #   Template A/B (h_out_per_step=2)               → 2
    #   Template C (cin=1, k=3, h_out_per_step=1)     → 1
    #   Template D (k=1, cin≤8, store_mode=2)         → 1
    #   Template E (k=1, cin>8, h_out_per_step=4)     → 4 (FSRCNN L1)
    #   Template F (k=3, cin>8, cout≤8, non-pixshuf)  → 2  (default)
    #   deformable conv                                → 4
    #   pixelshuffle (acc_mode=5, FSRCNN last_part)    → 128
    storer_step: int = 2
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
    # Hardware always processes 32-row spatial tiles regardless of model input height.
    tile_h = 32

    # Guide §3.1: deformable / bilinear uses 6-row line buffer; standard conv uses 4.
    if layer.deformable:
        line_rows = 6
        read_mode = 0
        bilinear = 1
        ky_outer = 3 if k == 3 else 1
        ic_inner = 2 if cin > 1 else 1
        h_out_per_step = 4
        load_total_num = max(1, tile_h // h_out_per_step)
        padding_num = 1
        acc_mode = 4        # pooling accumulate mode for deformable
        store_mode = 3      # deformable store mode
        lbr = 0             # golden: deformable layers use lbr=0
        cin_group = 4 if cin <= 8 else 8
        wpar = 0 if cout <= 8 else 1
        wt = 12  # bilinear WeightLoader transnum for deformable
        wl_lrs = 5          # WeightLoader.line_buffer_row_shift for deformable
        wl_ipc = 6          # WeightLoader.is_padding_col for deformable
        storer_step = 4     # golden: deformable DS base_addrs_res += 4 per cal_idx
    else:
        read_mode = 0
        bilinear = 0
        ky_outer = 1
        ic_inner = 1

        # ── H-step and cin-group templates ──────────────────────────────────
        padding_num = 1  # default; 1×1 conv templates override to 0
        if cin == 1 and k == 3:
            # Template C: single-channel 3×3 (FSRCNN first_part / UNet L0).
            h_out_per_step = 1
            cin_group = 1
            wpar = 2
            wt = 9
            line_rows = 3       # golden: 3 input rows per DL call for cin=1 k=3
            lbr = 0
            wl_lrs = 0
            wl_ipc = 3
            storer_step = 1     # golden: Template C DS += 1 (h_out_per_step=1)
        elif k == 1 and cin <= 8:
            # Template D: 1×1 conv with small cin (FSRCNN L10 style).
            h_out_per_step = 1
            cin_group = 1
            wpar = 2
            wt = cin
            line_rows = 2       # golden: 2 rows per DL for small-cin 1×1
            lbr = 0
            wl_lrs = 4
            wl_ipc = 0
            padding_num = 0     # 1×1 conv: no spatial padding needed
            storer_step = 1     # golden: Template D DS += 1 (h_out_per_step=1)
        elif k == 1 and cin > 8:
            # Template E: 1×1 conv with large cin (FSRCNN L1 style).
            h_out_per_step = 4
            cin_group = 8
            wpar = 0
            wt = 4              # golden: 4 weights per ic_group (not 1)
            line_rows = 4
            lbr = 3             # golden: lbr=3 for Template E
            wl_lrs = 0
            wl_ipc = 0
            padding_num = 0     # 1×1 conv: no spatial padding needed
            storer_step = 4     # golden FSRCNN L1: DS += 4 (h_out_per_step=4)
        elif k == 3 and cin > 8 and cout <= 8:
            # Template F: 3×3 conv with large cin but small cout (FSRCNN L11 style).
            h_out_per_step = 4
            ky_outer = 3
            cin_group = 8
            wpar = 0
            wt = 6
            line_rows = 4
            lbr = 3             # golden: lbr=3 for Template F
            wl_lrs = 2
            wl_ipc = 3
            # Default Template F step=4 (per golden's "一次算出4行" when no
            # pixelshuffle). plan_all() will override to 128 if pixelshuffle
            # (acc_mode=5) applies — matching FSRCNN last_part which stores
            # 4-channel output to fsrcnn_output_buffer (4 values per row).
            storer_step = 4
        else:
            # Template A/B: standard 3×3 conv, two output rows per outer step.
            # This is also the fall-through for any (cin, cout, k) combination
            # that didn't match a specialized template above. Warn so callers
            # can verify correctness — A/B parameters are only validated for
            # the UNet/FSRCNN standard 3×3 shapes.
            if not (k == 3):
                print(
                    f"[WARNING] tiling: layer {layer.idx} (op={layer.op}, "
                    f"cin={cin}, cout={cout}, k={k}) matched no specialized "
                    f"template — falling back to Template A/B. Verify tiling "
                    f"correctness for this layer."
                )
            h_out_per_step = 2
            cin_group = 8 if cin > 8 else (4 if cin > 1 else 1)
            wpar = 0 if cout <= 8 else 1
            wt = 9 if k == 3 else k * k
            line_rows = 4
            lbr = 1 if w_in <= 128 else 0
            wl_lrs = 1
            wl_ipc = 1
            storer_step = 2     # golden Template A/B: DS += 2 (h_out_per_step=2)

        load_total_num = max(1, tile_h // h_out_per_step)
        acc_mode = 0
        store_mode = 0

    if cout <= 4:
        cout_g = 4
    elif cout <= 8:
        cout_g = 8
    else:
        cout_g = min(32, max(8, (cout + 7) // 8 * 8))

    w_micro = _pick_w_micro_tile(w_in)
    macros = _macro_w_tiles(w_in)

    # Per-template QuantLoader parameters (matched to golden sr_inst per-layer values)
    if layer.deformable:
        q_mode = 5
        q_transnum = max(layer.cout, 8)
    elif cin == 1 and k == 3:           # Template C
        q_mode = 3
        q_transnum = cout
    elif k == 1 and cin <= 8:           # Template D
        q_mode = 3
        q_transnum = cout
    elif k == 1 and cin > 8:            # Template E
        q_mode = 5
        q_transnum = cout
    elif k == 3 and cin > 8 and cout <= 8:  # Template F
        q_mode = 5
        q_transnum = cout
    else:                               # Template A/B
        q_mode = 0
        q_transnum = 4

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
        quant_mode=q_mode,
        quant_transnum=q_transnum,
        tile_h=tile_h,
        wl_line_buffer_row_shift=wl_lrs,
        wl_is_padding_col=wl_ipc,
        storer_step=storer_step,
        notes="deformable" if layer.deformable else "standard conv",
    )


def _derive_acc_store_mode(layer: LayerDesc, layers: List[LayerDesc]) -> tuple:
    """Derive (acc_mode, store_mode) from layer op, activation, and position.

    Rules from docs/record.md Phase-6 analysis:
      offset_gen                               → (1, 1)
      deformable_conv2d (last)                 → (2, 1)
      deformable_conv2d (other)                → (4, 3)
      conv with prelu + next layer is pool/og  → (4, 3)  pool-while-store
      conv with prelu                          → (1, 2)
      conv with relu                           → (1, 1)
      final conv (no activation, last layer)   → (5, 1)
      otherwise                                → (0, 0)
    """
    if layer.op == "offset_gen":
        return 1, 1

    if layer.op == "deformable_conv2d":
        dconv_idxs = [L.idx for L in layers if L.op == "deformable_conv2d"]
        is_last_dconv = (layer.idx == dconv_idxs[-1]) if dconv_idxs else True
        if is_last_dconv:
            return 2, 1
        return 4, 3

    # Find successor layer
    next_layers = [L for L in layers if L.idx == layer.idx + 1]
    next_layer = next_layers[0] if next_layers else None

    act = getattr(layer, "activation", None)

    # Conv feeding an offset_gen needs simultaneous 2×2 pool-while-store.
    if act in ("relu", "prelu") and next_layer is not None and next_layer.op == "offset_gen":
        return 4, 3

    if act == "prelu":
        return 1, 2
    if act == "relu":
        return 1, 1

    # Final output layer: last conv with no activation
    conv_idxs = [L.idx for L in layers if L.op in ("conv2d", "deformable_conv2d")]
    if conv_idxs and layer.idx == conv_idxs[-1]:
        return 5, 1

    return 0, 0


def plan_all(layers: List[LayerDesc]) -> List[TilingPlan]:
    plans = [choose_tiling(L) for L in layers]
    for L, P in zip(layers, plans):
        acc, store = _derive_acc_store_mode(L, layers)
        P.acc_mode = acc
        P.store_mode = store
        # Last deformable conv uses quant_mode=7 (golden sr_codegen L9 pattern).
        if L.op == "deformable_conv2d" and acc == 2:
            P.quant_mode = 7
        # Pixelshuffle output (acc_mode=5) — used by FSRCNN last_part. The
        # DataStorer writes 4 values per row to fsrcnn_output_buffer, so the
        # per-iteration base_addrs_res stride is 128 (golden line 3659).
        if acc == 5:
            P.storer_step = 128
    return plans
