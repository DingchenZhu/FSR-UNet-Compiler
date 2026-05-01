"""Tiling decisions derived from docs/unet_fsrcnn_tiling_and_codegen_guide.md.

Applied as a separate stage between LayerDesc extraction and ISA emission.
The TilingPlan carries all parameters needed by the emitter — no magic numbers
should appear in emitter.py.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ir.layer_desc import LayerDesc


def _words_per_row(w_in: int) -> int:
    """DataLoader 64px-word granularity: max(1, ceil(w_in/64))."""
    return max(1, math.ceil(w_in / 64))

# Sentinel: use layer.h_in as the effective tile height (full-height processing).
# Pass to plan_all() / choose_tiling() for SD-UNet / full-image pipelines.
# Keep tile_h=32 (default) for FSRCNN tiled-32-row mode.
TILE_H_FULL = None


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
    # Group convolution support. For groups==1 (the default), all of the
    # following fields are inert and the emitter takes the standard non-group
    # path — FSRCNN and any other group=1 model are completely unaffected.
    # For groups>1, _apply_group_params() populates these from per-template
    # formulas that mirror sd_sr_codegen.py golden (conv6/7/8/10).
    group_count: int = 1            # equals layer.groups
    group_level1: int = 1           # outer group-loop iterations (conv7/8/10: 2)
    group_level2: int = 1           # inner group-loop iterations (conv6: g, conv10: 4)
    group_ql_in_level2: bool = False  # True → emit QuantLoader inside level2 loop
    # DL DataLoader bas_addr stride per (level1, level2) iteration (in 64px words).
    dl_level1_stride: int = 0
    dl_level2_stride: int = 0
    # DS DataStorer base_addrs_res stride per (level1, level2) iteration (in words).
    ds_level1_stride: int = 0
    ds_level2_stride: int = 0
    # SD-UNet inner-loop calibration. The standard FSRCNN templates use
    # cin_group as the inner ic loop (and ky_outer for ky). SD-UNet has more
    # complex patterns where:
    #   * h_out_per_step = 1 with explicit ky inner of 3 (some layers)
    #   * ic_inner × ky inner with cin_group = ic_inner (split cin)
    #   * oc_inner outer loop (L14/L16) — emits the body oc_inner times,
    #     incrementing DS base_addrs_res by ds_oc_stride per oc step.
    #   * ic_only inner (no ky) — used for some 1×1-style 3×3 layers (L17, L18)
    #     where ky is implicit in line buffer and only ic iterates.
    # Default: oc_inner=1 (no oc loop), ic_only=False (ky_outer drives).
    oc_inner: int = 1
    ds_oc_stride: int = 0
    ic_only_no_ky: bool = False
    # SD-UNet QuantLoader dispatch modes (Phase 20 P1 instruction count fix).
    # Default (both False): one QL emitted before all macro W tiles for the layer.
    # ql_per_macro=True: emit one QL per macro W tile (idx=1,2,21,22). Both
    #   tiles use the SAME quant_config_idx — no toggle between tiles.
    # ql_per_oc_iter=True: emit QL inside the oc_inner loop, one per oc step
    #   (idx=18,20). quant_config_idx is toggled BETWEEN oc iterations so the
    #   alternating QL/DS pairs share matching quant_config_idx values.
    # The two flags are mutually exclusive — at most one should be True.
    ql_per_macro: bool = False
    ql_per_oc_iter: bool = False
    # SD-UNet DepthToSpace transparent injection (Phase 18). When the conv
    # immediately precedes a DepthToSpace node, the hardware emits the pixel-
    # shuffle output via the DataStorer is_pixelshuffle=1 path with per-layer
    # specific field values (pixelshuffle_out_mode/transfer_num/stride/etc).
    # FSRCNN's last_part also has is_pixelshuffle=1 via the legacy
    # `acc_mode == 5` heuristic, which still works because both flags below
    # default to off — only the SD-UNet pre-DepthToSpace plans set
    # is_pixelshuffle=True with explicit field overrides.
    is_pixelshuffle: bool = False
    pixelshuffle_out_mode: int = 0
    pixelshuffle_transfer_num: int = 1   # golden DS.transfer_num for these layers
    pixelshuffle_stride: int = 0          # golden DS.stride for these layers
    pixelshuffle_acc_mode: int = 0        # golden DS.acc_mode for these layers
    pixelshuffle_store_mode: int = 0      # golden DS.store_mode for these layers
    # Pool-preceding conv (Phase 21). When the next layer is a pool2d node, the
    # DataStorer must also write the pooled result; this is signalled by
    # is_pooling=1 with a height-dependent pooling_out_mode. Distinct from the
    # deformable-conv pool path (store_mode==3, which sets pooling_out_mode=3).
    has_pool_output: bool = False    # True → DS must also write pooled result
    pool_output_mode: int = 0        # pooling_out_mode field in DS (0/1/2)
    # Phase 22: Pool result base addresses (DS.base_addr_pooling).
    #   pool_addr_start  — base_addr_pooling for the FIRST DS instruction of the
    #                      layer. Per golden references in sd_sr_codegen.py
    #                      (sd_inst()):
    #                        idx=2  conv1_2 (h=144) → 1152  (= 144*4*2)
    #                        idx=5  conv3   (h=72)  → 1728  (= 1152 + 72*8)
    #                        idx=8  conv5   (h=36)  → 2016  (= 1728 + 288)
    #                        idx=11 conv7   (h=18)  → 2016  (= same)
    #   pool_addr_stride — per-DS-pair (or per-DS) increment of base_addr_pooling.
    #                      For h=144 (idx=2): increments every DS step  (pair=1).
    #                      For h<=72 (idx=5/8/11): increments every 2 DS (pair=2).
    #   pool_addr_inc_period — number of consecutive DS instructions that share
    #                      the same base_addr_pooling. The hardware accumulates
    #                      2 conv-output rows into one pooled row, so for layers
    #                      whose DS step == 1 conv-row, every 2 DS share the
    #                      same pool address (period=2). For layers whose DS
    #                      step == 2 conv-rows already (h_out_per_step=2 →
    #                      conv1_2), every 1 DS gets its own pool address
    #                      (period=1).
    #   pool_addr_macro_stride — additive offset per macro W tile. Phase 23:
    #                      Golden conv1_2 right-half uses
    #                      base_addr_pooling_cur = pool_addr_start + 2, so
    #                      pool_addr_macro_stride = 2 for idx=2. Layers with a
    #                      single macro tile (idx=5/8/11) leave this at 0.
    #   pool_addr_group_stride — additive offset per group_level1 iteration
    #                      for group conv. Phase 23: idx=11 conv7 (g=8) uses
    #                      base_addr_pooling_cur = 2016 + group_level1_idx * 72.
    pool_addr_start: int = 0
    pool_addr_stride: int = 4
    pool_addr_inc_period: int = 1
    pool_addr_macro_stride: int = 0
    pool_addr_group_stride: int = 0
    # Phase 28 Fix 1: mask-store DataStorer pattern (SD-UNet idx=22 only).
    # When True, the DS for this layer is emitted with:
    #   is_mask=1
    #   is_new = 1 if cal_idx % 4 == 0 else 0
    #   transfer_num=0, store_mode=1, stride=0, acc_mode=0
    #   is_pixelshuffle=0, pixelshuffle_out_mode=0
    #   base_addrs_res increment ONLY when cal_idx % storer_increment_period
    #     == storer_increment_period - 1 (golden line 2274/2464:
    #     "if cal_idx % 4 == 3: base_addrs_res_cur += 2")
    #   right macro tile starts at ds_base + mask_macro_offset (1 in golden,
    #     not the usual tile_h * 4 = 144*4 = 576)
    # The is_mask flag bypasses the standard `is_pixshuffle_legacy` path that
    # `_derive_acc_store_mode` would otherwise force for the terminal conv
    # (acc_mode=5). For SD-UNet, the terminal conv writes to unet_output_reg
    # via the masked-store pattern, NOT FSRCNN-style pixelshuffle.
    is_mask: bool = False
    storer_increment_period: int = 1   # 4 for idx=22; 1 elsewhere (every cal_idx)
    mask_macro_offset: int = 0          # right-half DS base offset for mask layers
    # Phase 31: macro-tile pipelined producer. When True, BOTH macro tiles of
    # the layer write DataStorer base_addrs_res starting at 0 — i.e. the right
    # macro re-uses the left macro's address slot. This matches the archived
    # `pseudo_code_load_next_mid.txt` golden for L=1 (conv1_1, idx=1) and
    # L=17 (conv17, idx=21): those producers write to an INTERMEDIATE buffer
    # whose data is consumed by the immediately-next layer in macro-tile-
    # interleaved order before the right macro fires, so the slot can be
    # reused. The flag suppresses the default `tile_half_offset = tile_h * 4`
    # for macro_idx > 0. Set ONLY via `_UNET_IDX_OVERRIDE_TABLE` — must NOT
    # apply to layers that share a shape entry with non-pipelined siblings
    # (e.g. idx=2 conv1_2 has the same shape but does NOT use this pattern).
    same_base_for_macros: bool = False
    # Phase 31: pre-toggle acc_reg_idx at the start of the layer. The archived
    # `pseudo_code_load_next_mid.txt` golden interleaves conv11's back-half (5
    # DSes with DL.layer_idx=11) BETWEEN conv7 (L=8) and conv8 (L=9), shifting
    # the acc_reg_idx parity into L=9/L=10 DSes. Our compiler emits all of
    # conv11 contiguously after conv10 (L=10), so the parity at L=9 starts at
    # the wrong value. Setting `flip_acc_reg_idx_on_entry=True` for our L=9
    # (conv8, layer.idx=13) restores the golden parity for L=9 and L=10. L=11
    # multiset is unaffected (same 5 zeros + 5 ones either way), so the flip
    # does not propagate to subsequent layers.
    flip_acc_reg_idx_on_entry: bool = False
    # Phase 29: explicit DataStorer.stride override. The default branch of the
    # emitter falls back to `plan.tile_h`, which matches Template A/B (where
    # the DS writes one full output row spanning tile_h words). For SD-UNet
    # ky_outer>1 layers, golden writes a single accumulated value per cal_idx
    # with stride=0 (no per-row stride), and the emitter must emit that exact
    # value. When `ds_stride is None`, fall back to `plan.tile_h` for backward
    # compat with FSRCNN. When set, use the explicit value (typically 0 for
    # SD-UNet middle/decoder convs).
    ds_stride: Optional[int] = None
    # Phase 31: explicit DataStorer.transfer_num override. When None (default),
    # the emitter's `pix_transfer_num=1` rule applies for the standard non-
    # pixshuffle path. When set, the override pins transfer_num to that value
    # for the layer's DSes. Used for SD-UNet conv8 (golden L=9) which has
    # transfer_num=0 (sd_sr_codegen line 1383) despite not being a
    # pixelshuffle output.
    ds_transfer_num: Optional[int] = None
    # Phase 32: override transfer_num only for the LAST DS of each group
    # iteration (load_idx == load_total_num - 1). Used for conv11 (L11, g=2)
    # where sd_sr_codegen emits `1 if cal_idx < cal_total_num-1 else 0`
    # (line 1285 / 1578) — all but the final DS get transfer_num=1, the last
    # gets 0. Setting ds_last_transfer_num=0 replicates that pattern.
    ds_last_transfer_num: Optional[int] = None
    # Phase 30: explicit DataLoader bas_addr per-row advance overrides.
    # The emitter's default formula (2*(w_words-1) for padding, 2*w_words for
    # non-padding) is correct for layer 0 (SRAM read) but mismatches the golden
    # for SD-UNet layers 1..18 where the DL reads from an upstream buffer that
    # uses different stride conventions. When None (default), fall back to the
    # historic w_words formula — preserves FSRCNN parity. When set, the emitter
    # uses the supplied (pad, non_pad) advance values verbatim per row.
    dl_advance_pad: Optional[int] = None
    dl_advance_nopad: Optional[int] = None
    # Per-layer last-step DL transnum override. When not None, the LAST
    # load_idx step uses this value instead of line_buffer_rows. Used for
    # conv11 (h=18, h_out_per_step=4): last step has only 18%4=2 rows.
    last_step_transnum: Optional[int] = None
    # Weight bank slot index (0/1/2). The hardware provides 3 independent
    # weight SRAM slots; layers are partitioned across them in golden:
    #   Slot 0 (default): all FSRCNN layers; SD-UNet h≥72 encoder (L0-4),
    #                     conv11 (L11), all decoder h≥72 layers (L15-18).
    #   Slot 1: SD-UNet h=36 encoder (L5-6), decoder conv13/14 (L13-14).
    #   Slot 2: SD-UNet group convs — conv6/7/8/10 (L7-10), decoder conv12 (L12).
    # The emitter uses st.weight_bas_addr[plan.wl_slot] for all WL bas_addr
    # accesses and advances the correct slot after each layer.
    wl_slot: int = 0
    notes: str = ""


def _apply_group_params(plan: TilingPlan, layer: LayerDesc) -> None:
    """Populate group conv tiling fields from sd_sr_codegen.py golden formulas.

    The four supported patterns (verified against
    references/sd_sr_codegen.py lines 1016-1504):

      conv6  (g=2, cout==cin):    single group_idx loop, QL per group.
        DL = base + group_idx * 2;            DS = base + group_idx * 18 * 8
      conv7  (g=8, cout==cin):    level1 only, level2 unrolled in ic_load.
        DL = base + level1 * 18 * 8;          DS = base + level1 * 18 * 8
      conv8  (g=8, cout==cin):    level1 only, level2 unrolled in ic_load.
        DL = base + level1 * 9 * 8;           DS = base + level1 * 9 * 4
      conv10 (g=8, cout>cin):     true level1 × level2 nesting, QL per inner.
        DL = base + level1 * 9*4 + level2;    DS = base + level1*18*8 + level2*18*2

    For other group counts a generic fall-back is used (with a warning).
    """
    g = layer.groups
    plan.group_count = g

    if g == 8 and layer.cout > layer.cin:
        # conv10 pattern: true two-level loop, QL emitted inside level2.
        # Golden: dataloader_bas_addr = base + group_level1_idx*9*4 + group_level2_idx
        #         base_addrs_res      = base + group_level1_idx*18*8 + group_level2_idx*18*2
        plan.group_level1 = 2
        plan.group_level2 = 4
        plan.group_ql_in_level2 = True
        plan.dl_level1_stride = 9 * 4   # 36
        plan.dl_level2_stride = 1
        plan.ds_level1_stride = 18 * 8  # 144
        plan.ds_level2_stride = 18 * 2  # 36

    elif g == 8:
        # conv7 / conv8 pattern: level1 only, group_level2 unrolled inside the
        # cal_idx body (ic_load_num_per_cal=2, ky_load_num_per_cal=3 — NOT
        # visible to the QL/DL/DS loop scaffold). QL emitted once per level1.
        # Golden conv7: DL stride = 18*8, DS stride = 18*8
        # Golden conv8: DL stride = 9*8,  DS stride = 9*4
        # Distinguishing conv7 vs conv8: conv7 has w_in==32, conv8 has w_in==16.
        plan.group_level1 = 2
        plan.group_level2 = 1
        plan.group_ql_in_level2 = False
        if layer.w_in >= 32:
            # conv7: DL += level1*18*8, DS += level1*18*8 (golden line 1134, 1135)
            plan.dl_level1_stride = 18 * 8  # 144
            plan.ds_level1_stride = 18 * 8  # 144
        else:
            # conv8: DL += level1*9*8, DS += level1*9*4 (golden line 1332, 1333)
            plan.dl_level1_stride = 9 * 8   # 72
            plan.ds_level1_stride = 9 * 4   # 36
        plan.dl_level2_stride = 0
        plan.ds_level2_stride = 0

    elif g == 2:
        # conv6 pattern: single group_idx loop (modeled here as level2 with
        # level1=1). QL emitted per group → group_ql_in_level2=True.
        # Golden line 1036, 1037:
        #   dataloader_bas_addr = base + group_idx * 2
        #   base_addrs_res      = base + group_idx * 18 * 8
        plan.group_level1 = 1
        plan.group_level2 = g
        plan.group_ql_in_level2 = True
        plan.dl_level1_stride = 0
        plan.dl_level2_stride = 2
        plan.ds_level1_stride = 0
        plan.ds_level2_stride = 18 * 8  # 144

    else:
        warnings.warn(
            f"Group conv g={g} (layer {layer.idx}) not explicitly calibrated; "
            f"using generic single-loop fall-back."
        )
        cin_g = layer.cin // g
        plan.group_level1 = 1
        plan.group_level2 = g
        plan.group_ql_in_level2 = True
        plan.dl_level1_stride = 0
        plan.dl_level2_stride = cin_g * _words_per_row(layer.w_in) // max(1, plan.cin_group)
        plan.ds_level1_stride = 0
        plan.ds_level2_stride = plan.load_total_num * plan.storer_step


# ─────────────────────────────────────────────────────────────────────────────
# SD-UNet per-layer parameter overrides (Phase 15 calibration).
#
# Keyed by the LayerDesc shape signature (h_in, w_in, cin, cout, k, groups).
# Values were derived directly from references/sd_sr_codegen.py sd_inst() — see
# docs/record.md "Phase 15" section for the per-layer cross-reference.
#
# These overrides apply only in full-height streaming mode (tile_h=None), so
# FSRCNN tiled-32 mode is completely untouched. The override produces a tuple
# of (h_out_per_step, cin_group, ky_outer, ic_inner_only_flag, oc_inner,
#  weight_transnum, weight_parall_mode, line_buffer_reshape, line_buffer_rows,
#  wl_lrs, wl_ipc, quant_mode, quant_transnum, ds_storer_step, ds_oc_stride,
#  acc_mode, store_mode, padding_num).
#
# acc_mode/store_mode here are PROVISIONAL — they get overridden by
# _derive_acc_store_mode() in plan_all().  But we set them to the golden values
# here to make tests/golden_diff stable when plan_all() agrees.
# ─────────────────────────────────────────────────────────────────────────────

_UNET_OVERRIDE_KEY = "unet_override"

# (h, w, cin, cout, k, g) → dict of override fields
_UNET_LAYER_TABLE = {
    # Encoder block 1 (h=144, w=256, conv1_*): standard A/B works as-is, but we
    # also pin storer_step=2 for clarity. Override is a no-op since defaults match.
    (144, 256,  1,  4, 3, 1): {  # golden L0 conv1: cin=1→cout=32 (32=4*8)
        "h_out_per_step": 2, "cin_group": 1, "ky_outer": 1,
        "weight_transnum_base": 9, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 1, "wl_ipc": 1,
        "quant_mode": 0, "quant_transnum": 4,
        "storer_step": 2, "padding_num": 1,
    },
    (144, 256,  4,  4, 3, 1): {  # golden L1/L2 conv1_1 / conv1_2
        "h_out_per_step": 2, "cin_group": 4, "ky_outer": 1,
        "weight_transnum_base": 9, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 1, "wl_ipc": 1,
        "quant_mode": 0, "quant_transnum": 4,
        "storer_step": 2, "padding_num": 1,
        # Golden conv1_1 / conv1_2 emit ONE QL per macro W tile (the archived
        # sd_inst dump issues 2 QLs for each of these layers). ql_per_macro=True
        # is required to match the 17155-instruction count.
        "ql_per_macro": True,
    },
    # Encoder block 2 (h=72, w=128).
    # Phase 27: storer_step set to per-template golden values. For SD-UNet
    # h_out_per_step=1 layers with cout > 4, the conv writes one output row
    # per cal_idx and the row spans `cout/word_lane` words; the per-DS
    # increment matches the cout-to-storage mapping (8 words/row for cout
    # ranges 8..16 mapped to 8 words via word-lane=8 packing in
    # sd_sr_codegen.py). Verified against golden lines noted per layer.
    (72, 128,  4,  8, 3, 1): {   # golden L3 conv2 (32→64) — line 718 +=8
        "h_out_per_step": 1, "cin_group": 1, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 0,
        "line_buffer_reshape": 1, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 1,
        "quant_mode": 1, "quant_transnum": 8,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L3 DS uses (acc=1, store=1, stride=0) — Template
        # "ky-software-loop" pattern (line_buffer_reshape=1 implies the
        # accumulated output is written as a single value per cal_idx, no
        # per-row stride). See sd_sr_codegen.py line 705-708.
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
    },
    (72, 128,  8,  8, 3, 1): {   # golden L4 conv3 (64→64) — line 812 +=8
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 1,
        "quant_mode": 1, "quant_transnum": 8,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L4 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
    },
    # Encoder block 3 (h=36, w=64).
    (36, 64,  8, 16, 3, 1): {    # golden L5 conv4 (64→128) — line 907 +=8
        "h_out_per_step": 1, "cin_group": 1, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 1,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 3, "wl_ipc": 2,
        "quant_mode": 2, "quant_transnum": 16,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L5 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
        "wl_slot": 1,
    },
    (36, 64, 16, 16, 3, 1): {    # golden L6 conv5 (128→128) — line 1001 +=8
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 1,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 3, "wl_ipc": 2,
        "quant_mode": 2, "quant_transnum": 16,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L6 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
        "wl_slot": 1,
    },
    # Encoder block 4 (h=18, w=32) — group conv. Patterns delegate to
    # _apply_group_params(), but we also override the weight params here.
    (18, 32, 16, 64, 3, 2): {    # golden L7 conv6 (128→128 g=2) — line 1098 +=8
        "h_out_per_step": 1, "cin_group": 1, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 2,
        "line_buffer_reshape": 0, "line_buffer_rows": 2,
        "wl_lrs": 4, "wl_ipc": 3,
        "quant_mode": 3, "quant_transnum": 32,
        "storer_step": 8, "padding_num": 1,
        # Group params: conv6-style (g=2): level1=1, level2=2, ql per level2.
        # Already handled by _apply_group_params; we just set load_total here.
        # Phase 29: golden L7 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
        "wl_slot": 2,
    },
    (18, 32, 64, 64, 3, 8): {    # golden L8 conv7 (128→128 g=8) — line 1197 +=1 (h-cont.)
        # Pre-pool bottleneck conv with "h continuous" pixelshuffle-prep
        # output layout (golden comment "此时是h连续"). DS step intentionally
        # 1 even though h_out_per_step=1 — each DS writes one column/row
        # transposed.
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 2,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 3,
        "quant_mode": 3, "quant_transnum": 32,
        "storer_step": 1, "padding_num": 1,
        # Phase 29: golden L8 DS uses (acc=1, store=2, stride=18) — h-continuous
        # write pattern. See sd_sr_codegen.py line 1190-1200.
        "acc_mode": 1, "store_mode": 2, "ds_stride": 18,
        "wl_slot": 2,
    },
    # Encoder block 5 (h=9, w=16) — bottleneck g=8.
    (9, 16, 64, 64, 3, 8): {     # golden L9 conv8 (128→128 g=8) — line 1374-1395
        # Phase 31: archived golden DS (sd_sr_codegen line 1382-1385) uses
        # (acc_mode=3, transfer_num=0, store_mode=1, stride=0). Without these
        # explicit overrides, _derive_acc_store_mode falls back to (0,0)
        # because conv8 is followed by conv10 (no special activation), and
        # the standard "stride=tile_h=9, transfer_num=1" defaults take over.
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 2,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 4,
        "quant_mode": 4, "quant_transnum": 32,
        "storer_step": 4, "padding_num": 1,
        "acc_mode": 3, "store_mode": 1, "ds_stride": 0,
        "ds_transfer_num": 0,
        "wl_slot": 2,
    },
    (9, 16, 64, 256, 3, 8): {    # golden L10 conv10 (128→256 g=8) — pre-DTS conv1
        # NOTE: ic_inner unrolled into ic_load_num_per_cal=1 (just ky), per
        # golden code line 1426-1428. Each level1×level2 emits 9*3=27 DL/WL.
        # DepthToSpace fold-in (Phase 18, golden QL.layer_idx=11):
        #   pix_out_mode=0, acc_mode=3, store_mode=0, transfer_num=0, stride=18
        # Phase 27: storer_step=2 per golden line 1493 (DS += 2 inside the
        # group_level2 inner loop; pre-DTS pixelshuffle output).
        # Phase 31: archived golden DL.transnum=1 (line 1448) — overrides the
        # default `line_buffer_rows=4` for the DL `transnum` field. WL still
        # uses 24 (line 1464). This is a per-field divergence: the DL reads
        # 1 word at a time (matching cin=1 unrolled) while WL reads the full
        # 24-byte weight block per step.
        "is_pixelshuffle": True,
        "pixelshuffle_out_mode": 0,
        "pixelshuffle_acc_mode": 3,
        "pixelshuffle_store_mode": 0,
        "pixelshuffle_transfer_num": 0,
        "pixelshuffle_stride": 18,
        "h_out_per_step": 1, "cin_group": 1, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 2,
        "line_buffer_reshape": 0, "line_buffer_rows": 1,
        # Phase 31: archived golden has WL.line_buffer_row_shift=5 (vs
        # sd_sr_codegen.py line 1459's wlrs=6). Matching the archive.
        "wl_lrs": 5, "wl_ipc": 4,
        "quant_mode": 4, "quant_transnum": 32,
        "storer_step": 2, "padding_num": 1,
        "wl_slot": 2,
    },
    # Decoder block 1 (h=18, w=32, after deconv1 = DepthToSpace1).
    (18, 32, 128, 16, 3, 2): {   # golden L11 conv11 (128→16 g=2) — line 1297/1590 +=8
        # golden L11 "conv11 (128,16,3,2)": ic_load=16, h_out=4, cal_total=5,
        # line_buffer_reshape=3, wpar=0, wt=12, wlrs=0, acc=2, store=1, stride=0.
        # Last cal_idx uses only 18%4=2 rows → last_step_transnum=2.
        "h_out_per_step": 4, "cin_group": 16, "ky_outer": 3,
        "load_total_num": 5,
        "weight_transnum_base": 12, "weight_parall_mode": 0,
        "line_buffer_reshape": 3, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 3,
        "quant_mode": 7, "quant_transnum": 8,
        "storer_step": 8, "padding_num": 1,
        "acc_mode": 2, "store_mode": 1, "ds_stride": 0,
        "last_step_transnum": 2,
        # Phase 32: golden emits transfer_num=0 for last DS of each group
        # (sd_sr_codegen line 1285/1578: 1 if cal_idx < cal_total-1 else 0).
        "ds_last_transfer_num": 0,
        # DS group base: group0=36, group1=0 → ds_level2_stride=36, base=0.
        "ds_level2_stride": 36,
    },
    # NOTE: shape (18,32,16,64,3,2) is shared by encoder conv6 (golden L7)
    # AND decoder conv12 (golden L13). The encoder version is keyed above; the
    # decoder version (cin_group=4, weight_parall_mode=1, qm=2, etc.) is
    # disambiguated via _UNET_IDX_OVERRIDE_TABLE keyed by layer.idx==16.
    # Decoder block 2 (h=36, w=64, after deconv2).
    (36, 64, 32, 16, 3, 1): {    # golden L13 conv13 — line 1783 +=8
        # Phase 20 P1: structural alignment — cin_group 2→4, oc_inner removed.
        # The oc_inner=2 pattern moved to the (36,64,16,32) entry below
        # (golden L14 conv14 with per-oc QL).
        "h_out_per_step": 1, "cin_group": 4, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 1,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 3, "wl_ipc": 2,
        "quant_mode": 2, "quant_transnum": 16,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L13 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
        "wl_slot": 1,
    },
    (36, 64, 16, 32, 3, 1): {    # golden L14 conv14 part 2 (oc=2) — pre-DTS conv3
        # DepthToSpace fold-in (Phase 18, golden QL.layer_idx=15):
        #   pix_out_mode=2, acc_mode=1, store_mode=1, transfer_num=1, stride=0
        # Phase 27: storer_step=16 per golden line 1886 (per-cal_idx +=16
        # within oc inner loop; ds_oc_stride=8 supplies the +oc*8 offset).
        # Phase 31: archived golden QL.transnum=16, quant_mode=2 (sd_sr_codegen
        # lines 1801, 1819) — earlier table had transnum=8/qm=1 mismatching.
        "is_pixelshuffle": True,
        "pixelshuffle_out_mode": 2,
        "pixelshuffle_acc_mode": 1,
        "pixelshuffle_store_mode": 1,
        "pixelshuffle_transfer_num": 1,
        "pixelshuffle_stride": 0,
        # Phase 20 P1: structural alignment — cin_group 4→2, add oc_inner=2
        # with per-oc QL. The two oc iterations alternate quant_config_idx.
        # Weight params mirror the h=36 encoder (conv4/5): wpar=1, wt=24, lrs=3, ipc=2.
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 1,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 3, "wl_ipc": 2,
        "quant_mode": 2, "quant_transnum": 16,
        "storer_step": 16, "padding_num": 1,
        "oc_inner": 2, "ds_oc_stride": 8, "ql_per_oc_iter": True,
        "wl_slot": 1,
    },
    # Decoder block 3 (h=72, w=128).
    (72, 128, 16, 8, 3, 1): {    # golden L15 conv15 — line 1898 (sd_inst)
        # Phase 20 P1: structural alignment — cin_group 2→4, oc_inner removed.
        # The oc_inner=2 pattern moved to the (72,128,8,16) entry below
        # (golden L16 conv16 with per-oc QL).
        # Phase 31: archived golden QL.quant_mode=1 (line 1900).
        "h_out_per_step": 1, "cin_group": 4, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 1,
        "quant_mode": 1, "quant_transnum": 8,
        "storer_step": 8, "padding_num": 1,
        # Phase 29: golden L15 DS uses (acc=1, store=1, stride=0).
        "acc_mode": 1, "store_mode": 1, "ds_stride": 0,
    },
    (72, 128, 8, 16, 3, 1): {    # decoder conv16 (8→16, oc=2) — pre-DTS conv4
        # DepthToSpace fold-in (Phase 18, golden QL.layer_idx=17):
        #   pix_out_mode=2, acc_mode=6, store_mode=2, transfer_num=1, stride=144
        # Phase 27: storer_step=2 per golden line 2084 (per-cal_idx += 2
        # within oc inner loop; ds_oc_stride=1 supplies the +oc*1 offset).
        # Phase 31: archived golden QL.quant_mode=6 (mirrors pixelshuffle_acc_mode=6;
        # confirmed against the (8,16) decoder pre-DTS in golden L=16 = 2 QLs of
        # (transnum=8, quant_mode=6)).
        "is_pixelshuffle": True,
        "pixelshuffle_out_mode": 2,
        "pixelshuffle_acc_mode": 6,
        "pixelshuffle_store_mode": 2,
        "pixelshuffle_transfer_num": 1,
        "pixelshuffle_stride": 144,
        # Phase 20 P1: structural alignment — cin_group 4→2, add oc_inner=2
        # with per-oc QL (alternates quant_config_idx between oc iterations).
        "h_out_per_step": 1, "cin_group": 2, "ky_outer": 3,
        "weight_transnum_base": 12, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 0, "wl_ipc": 1,
        "quant_mode": 6, "quant_transnum": 8,
        "storer_step": 2, "padding_num": 1,
        "oc_inner": 2, "ds_oc_stride": 1, "ql_per_oc_iter": True,
        # wl_slot=0 (default): slot 0 resumes after encoder bottleneck
    },
    # Final decoder layers (h=144, w=256) — fused L17/L18 pattern in golden.
    # In our IR these are two separate convs; we emit the full ic-loop pattern
    # for each. This produces the right total count even though golden fuses
    # them into a single pass per macro tile.
    (144, 256, 8, 4, 3, 1): {    # golden L17 conv17 (64→32) — ic_only=True
        "h_out_per_step": 2, "cin_group": 8, "ky_outer": 1,
        "weight_transnum_base": 9, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 1, "wl_ipc": 1,
        "quant_mode": 0, "quant_transnum": 4,
        "storer_step": 2, "padding_num": 1,
        "ic_only_no_ky": True,
        # Phase 20 P1: emit one QL per macro W tile (golden L17 issues 2 QLs).
        "ql_per_macro": True,
    },
    (144, 256, 4, 1, 3, 1): {    # golden L18 conv18 (32→4) — ic_only=True
        "h_out_per_step": 2, "cin_group": 4, "ky_outer": 1,
        "weight_transnum_base": 9, "weight_parall_mode": 0,
        "line_buffer_reshape": 0, "line_buffer_rows": 4,
        "wl_lrs": 1, "wl_ipc": 1,
        "quant_mode": 0, "quant_transnum": 1,
        "storer_step": 2, "padding_num": 1,
        "ic_only_no_ky": True,
        # Phase 20 P1: emit one QL per macro W tile (golden L18 issues 2 QLs).
        "ql_per_macro": True,
        # Phase 28 Fix 1: SD-UNet terminal layer uses the masked-store DS
        # pattern (golden L18 line 2253-2275 left, line 2443-2465 right):
        #   is_mask=1, transfer_num=0, store_mode=1, stride=0, acc_mode=0
        #   is_new = 1 if cal_idx % 4 == 0 else 0
        #   base_addrs_res starts at 0 (left macro) / 1 (right macro)
        #   increments by 2 only when cal_idx % 4 == 3
        # The non-standard +1 right-macro offset (vs the usual tile_h*4=576)
        # is the "give right-half its own slot" hardware quirk noted in the
        # golden's per-cal_idx-mod-4 comment ("交给reg自己判断，+2是给右半边留空间").
        "is_mask": True,
        "storer_step": 2,
        "storer_increment_period": 4,
        "mask_macro_offset": 1,
    },
}


# Disambiguating override table: when two non-adjacent layers share the same
# shape signature (e.g. encoder conv6 and decoder conv12 are both
# (18,32,16,64,3,2)) we key by the LayerDesc.idx instead. The runtime first
# looks up by idx, then falls back to shape.
_UNET_IDX_OVERRIDE_TABLE = {
    # Phase 22+23: Pool-preceding conv layers (idx=2,5,8,11) need explicit
    # base_addr_pooling values. Detection of has_pool_output happens in
    # plan_all() based on the successor being pool2d — these idx-keyed
    # entries supply ONLY the pool address fields and inherit everything
    # else from the shape-keyed _UNET_LAYER_TABLE entry. The merge happens
    # in _try_unet_override(), which now combines the two override sources.
    #
    # Golden derivation (sd_sr_codegen.py sd_inst()):
    #
    #   idx=2  (conv1_2, h=144, w=256, h_out_per_step=2):
    #          left half:  base = 1152, stride=4 every cal_idx (line 549-550)
    #          right half: base = 1152 + 2 = 1154, stride=4 every cal_idx
    #          → pool_addr_start=1152, pool_addr_stride=4, inc_period=1,
    #            pool_addr_macro_stride=2.
    #
    #   idx=5  (conv3, h=72, w=128, h_out_per_step=1):
    #          base = 1728; pool_addr_pooling_cur += 4 only when cal_idx%2==1
    #          (line 813-814) → inc_period=2, stride=4.
    #
    #   idx=8  (conv5, h=36, w=64, h_out_per_step=1):
    #          base = 2016; same += 4 if cal_idx%2==1 pattern (line 1002-1003)
    #          → inc_period=2, stride=4.
    #
    #   idx=11 (conv7, h=18, w=32, h_out_per_step=1, g=8):
    #          base = 2016 + group_level1_idx * 72 (line 1136),
    #          += 8 when cal_idx%2==1 (line 1198-1199)
    #          → inc_period=2, stride=8, pool_addr_group_stride=72.
    # Phase 31: archived golden L=1 (conv1_1) and L=17 (conv17) emit BOTH
    # macro tiles' DSes at base_addrs_res starting from 0 — the right macro
    # re-uses the left macro's slot. This is a macro-tile-pipelined producer
    # whose intermediate buffer is consumed by the next layer (idx=2 conv1_2 /
    # idx=22 conv18) before the right macro fires. The same-shape sibling
    # (idx=2 / idx=22) does NOT use this pattern, so the override is idx-keyed.
    1:  {"same_base_for_macros": True},
    21: {"same_base_for_macros": True},
    # Phase 31: layer.idx=13 (conv8) — pre-toggle acc_reg_idx to align with
    # archived golden's parity (conv11 back-half emits 5 DSes between L=8 and
    # L=9 in golden's interleaved schedule, shifting the L=9/L=10 reg_out_idx
    # by 1; our linear emit doesn't reproduce that interleave). Pair with the
    # idx=15 flip below so the parity returns to baseline before L=12.
    13: {"flip_acc_reg_idx_on_entry": True},
    # Phase 31: layer.idx=15 (conv11) — pre-toggle acc_reg_idx again. Golden's
    # conv11 split: 5 DSes before L=9 (toggles parity to 1) and 5 after L=10
    # (toggles parity back to 0). Our compiler emits all 10 contiguously, so
    # without this second flip, L=11 starts at the post-flip value (1) and
    # L=12 inherits 1 instead of golden's 0. The flip restores parity at
    # L=11's entry — its 10 DSes keep parity 0 after the layer, which is
    # exactly golden's post-conv11-front-half state.
    15: {"flip_acc_reg_idx_on_entry": True},
    2:  {"pool_addr_start": 1152, "pool_addr_stride": 4,
         "pool_addr_inc_period": 1, "pool_addr_macro_stride": 2},
    5:  {"pool_addr_start": 1728, "pool_addr_stride": 4,
         "pool_addr_inc_period": 2},
    8:  {"pool_addr_start": 2016, "pool_addr_stride": 4,
         "pool_addr_inc_period": 2},
    11: {"pool_addr_start": 2016, "pool_addr_stride": 8,
         "pool_addr_inc_period": 2, "pool_addr_group_stride": 72},
    # our idx 16 = decoder conv12 (cin=16, cout=64, g=2, h=18, w=32) →
    # golden L13. Distinguish from idx 10 (= encoder conv6, golden L7) which
    # has the same shape but different parameters (wpar=2 vs 1, wlrs=4 vs 3,
    # ipc=3 vs 2, qm=3 vs 2, qtn=32 vs 16).
    16: {
        # decoder conv12 — pre-DTS conv2 (golden QL.layer_idx=13):
        #   pix_out_mode=1, acc_mode=1, store_mode=3, transfer_num=1, stride=8
        # Phase 27: storer_step=16 per golden line 1684 (DS += 16 per cal_idx
        # for the per-group inner loop; ds_level2_stride=4 supplies +g*4 base
        # for each group, see _UNET_GROUP_OVERRIDE_TABLE).
        # Phase 31: realign WL/QL/DL fields with archived golden L=12
        # (sd_sr_codegen.py lines 1604-1670):
        #   DL.transnum=2 (line 1639), WL line_buffer_row_shift=4 (line 1650),
        #   WL is_padding_col=3 (line 1652), WL weight_parall_mode=2 (line
        #   1653), QL transnum=32 (line 1611), QL quant_mode=3 (line 1606).
        "is_pixelshuffle": True,
        "pixelshuffle_out_mode": 1,
        "pixelshuffle_acc_mode": 1,
        "pixelshuffle_store_mode": 3,
        "pixelshuffle_transfer_num": 1,
        "pixelshuffle_stride": 8,
        # Phase 20 P1: golden L12 "conv12 (16,64,3,2)" has ic_load_num_per_cal=1
        # (no inner ic loop, only ky). cin_group 4→1.
        "h_out_per_step": 1, "cin_group": 1, "ky_outer": 3,
        "weight_transnum_base": 24, "weight_parall_mode": 2,
        "line_buffer_reshape": 0, "line_buffer_rows": 2,
        "wl_lrs": 4, "wl_ipc": 3,
        "quant_mode": 3, "quant_transnum": 32,
        "storer_step": 16, "padding_num": 1,
        # Phase 28 Fix 3: conv12 group conv level2 strides differ from
        # encoder conv6 (idx=10). Golden L13 line 1622-1623:
        #   dataloadermanager.bas_addr_cur = 0 + group_idx * 18*2  → dl_lvl2 = 36
        #   datastorermanager.base_addrs_res_cur = 2016 + group_idx * 4 → ds_lvl2 = 4
        # The default conv6 g=2 pattern in _apply_group_params() sets
        # dl_level2_stride=2 and ds_level2_stride=144, which is correct for
        # the encoder layer but wrong for this decoder layer. The override
        # below replaces those values; _apply_group_params runs FIRST and the
        # override is applied AFTER (see choose_tiling: override loop is the
        # last thing before _apply_group_params, so we instead intercept by
        # asserting the override-time precedence — see Note in choose_tiling).
        "dl_level2_stride": 36,
        "ds_level2_stride": 4,
        "wl_slot": 2,
    },
}


def _try_unet_override(layer: LayerDesc) -> Optional[dict]:
    """Look up SD-UNet per-layer override.

    Resolution order:
      1. Start from shape signature (h, w, cin, cout, k, g) → _UNET_LAYER_TABLE.
      2. Overlay LayerDesc.idx → _UNET_IDX_OVERRIDE_TABLE (idx-keyed entries
         take priority — they disambiguate same-shape but semantically-
         different layers, and they may also carry SUPPLEMENTARY fields that
         the shape entry doesn't define (e.g. Phase 22 pool_addr_start)).

    Returns None only when both lookups miss.
    """
    key = (layer.h_in, layer.w_in, layer.cin, layer.cout, layer.k_h, layer.groups)
    shape_ov = _UNET_LAYER_TABLE.get(key)
    idx_ov = _UNET_IDX_OVERRIDE_TABLE.get(layer.idx)
    if shape_ov is None and idx_ov is None:
        return None
    merged: dict = {}
    if shape_ov:
        merged.update(shape_ov)
    if idx_ov:
        merged.update(idx_ov)   # idx-keyed wins on conflict
    return merged


def choose_tiling(layer: LayerDesc, tile_h: Optional[int] = 32) -> TilingPlan:
    """Map LayerDesc → TilingPlan using documented UNet/FSRCNN templates.

    tile_h: hardware spatial tile height.
      32    → FSRCNN tiled mode (default, preserves PERFECT FSRCNN match).
      None  → full-height mode for SD-UNet: effective tile = layer.h_in.
               Template C (cin=1 k=3 tiled) is disabled; falls to Template A/B.
    """

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
    # effective_tile_h: 32 for FSRCNN tiled mode; h_in for SD-UNet full-height mode.
    effective_tile_h = tile_h if tile_h is not None else h_in

    # Guide §3.1: deformable / bilinear uses 6-row line buffer; standard conv uses 4.
    if layer.deformable:
        line_rows = 6
        read_mode = 0
        bilinear = 1
        ky_outer = 3 if k == 3 else 1
        ic_inner = 2 if cin > 1 else 1
        h_out_per_step = 4
        load_total_num = max(1, effective_tile_h // h_out_per_step)
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
        if cin == 1 and k == 3 and effective_tile_h < h_in:
            # Template C: single-channel 3×3 TILED (FSRCNN first_part only).
            # Disabled in full-height mode (effective_tile_h == h_in) — SD-UNet's
            # cin=1 layers fall through to Template A/B (transnum=4, lrs=1).
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

        load_total_num = max(1, effective_tile_h // h_out_per_step)
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
    elif cin == 1 and k == 3 and effective_tile_h < h_in:  # Template C (tiled only)
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

    plan = TilingPlan(
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
        tile_h=effective_tile_h,
        wl_line_buffer_row_shift=wl_lrs,
        wl_is_padding_col=wl_ipc,
        storer_step=storer_step,
        notes="deformable" if layer.deformable else "standard conv",
    )

    # Group-conv tiling: populate plan.group_* and plan.{dl,ds}_level{1,2}_stride
    # with the per-template defaults BEFORE applying the SD-UNet override. This
    # lets per-idx overrides in `_UNET_IDX_OVERRIDE_TABLE` selectively replace
    # specific group fields (e.g. conv12 idx=16 needs dl_level2_stride=36 and
    # ds_level2_stride=4 to override the conv6-g=2 defaults).
    if layer.groups > 1:
        _apply_group_params(plan, layer)

    # SD-UNet calibration: in full-height streaming mode (tile_h=None), apply
    # per-shape overrides that mirror the golden sd_sr_codegen.py inner-loop
    # structure. Other modes (FSRCNN tiled-32) skip this entirely.
    if tile_h is None and not layer.deformable:
        ov = _try_unet_override(layer)
        if ov is not None:
            for k, v in ov.items():
                if k == "wl_lrs":
                    plan.wl_line_buffer_row_shift = v
                elif k == "wl_ipc":
                    plan.wl_is_padding_col = v
                elif hasattr(plan, k):
                    setattr(plan, k, v)
            # Recompute load_total_num from the overridden h_out_per_step
            # UNLESS the override explicitly sets it (e.g. idx=15 conv11
            # needs load_total_num=5 = ceil(18/4), not floor=4).
            if "load_total_num" not in ov:
                plan.load_total_num = max(1, effective_tile_h // plan.h_out_per_step)
            plan.notes = (
                f"SD-UNet override: h={layer.h_in} w={layer.w_in} "
                f"cin={layer.cin} cout={layer.cout} g={layer.groups}"
            )
    return plan


def _derive_acc_store_mode(layer: LayerDesc, layers: List[LayerDesc]) -> tuple:
    """Derive (acc_mode, store_mode) from layer op, activation, and position.

    Rules from docs/record.md Phase-6 analysis:
      offset_gen                               → (1, 1)
      deformable_conv2d (last)                 → (2, 1)
      deformable_conv2d (other)                → (4, 3)
      conv with prelu + next layer is pool/og  → (4, 3)  pool-while-store
      conv with prelu                          → (1, 2)
      conv with relu                           → (0, 0)  standard accumulation
      final conv (no activation, last layer)   → (5, 1)
      otherwise                                → (0, 0)

    relu vs prelu distinction: prelu requires hardware special accumulation (acc_mode=1)
    to track negative pre-activation values; relu clips at 0 and uses standard
    acc_mode=0. Confirmed by SD-UNet golden (relu layers → acc_mode=0) and FSRCNN
    golden (prelu layers → acc_mode=1).
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
    # relu → standard (0,0); hardware clips in post-processing, no special accumulation.
    # (prelu → (1,2) above; relu falls through to (0,0) default.)

    # Final output layer: last conv with no activation
    conv_idxs = [L.idx for L in layers if L.op in ("conv2d", "deformable_conv2d")]
    if conv_idxs and layer.idx == conv_idxs[-1]:
        return 5, 1

    return 0, 0


def plan_all(layers: List[LayerDesc], tile_h: Optional[int] = 32) -> List[TilingPlan]:
    plans = [choose_tiling(L, tile_h=tile_h) for L in layers]
    # Phase 27: detect SD-UNet topology (any layer carries skip_sources). The
    # acc=5 → storer_step=128 rule below is FSRCNN-specific (its last_part
    # writes 4 values per row to fsrcnn_output_buffer, so DS step=128); for
    # SD-UNet, the last conv (idx=22) goes to unet_output_reg via a masked-
    # store pattern that already has storer_step=2 set in its override entry.
    is_sd_unet = any(getattr(L, "skip_sources", None) for L in layers)
    for L, P in zip(layers, plans):
        acc, store = _derive_acc_store_mode(L, layers)
        P.acc_mode = acc
        P.store_mode = store
        # Phase 29: SD-UNet ky-software-loop layers carry per-shape DS field
        # overrides in `_UNET_LAYER_TABLE` (acc_mode/store_mode/ds_stride).
        # `_derive_acc_store_mode` would clobber them with the relu→(0,0)
        # default; re-apply the override here to restore the golden values.
        # Only do this for SD-UNet topology to avoid disturbing FSRCNN.
        if is_sd_unet:
            ov = _try_unet_override(L)
            if ov is not None:
                if "acc_mode" in ov:
                    P.acc_mode = ov["acc_mode"]
                if "store_mode" in ov:
                    P.store_mode = ov["store_mode"]
                if "ds_stride" in ov:
                    P.ds_stride = ov["ds_stride"]
                # Phase 31: re-apply ds_transfer_num after _derive_acc_store_mode
                # so the override sticks (initial choose_tiling override fires
                # before derive runs, but derive only changes acc/store; safe to
                # repeat the assignment for clarity).
                if "ds_transfer_num" in ov:
                    P.ds_transfer_num = ov["ds_transfer_num"]
                if "ds_last_transfer_num" in ov:
                    P.ds_last_transfer_num = ov["ds_last_transfer_num"]
        # Last deformable conv uses quant_mode=7 (golden sr_codegen L9 pattern).
        if L.op == "deformable_conv2d" and acc == 2:
            P.quant_mode = 7
        # Pixelshuffle output (acc_mode=5) — used by FSRCNN last_part. The
        # DataStorer writes 4 values per row to fsrcnn_output_buffer, so the
        # per-iteration base_addrs_res stride is 128 (golden line 3659).
        # SD-UNet's terminal layer (idx=22, acc=5 by default heuristic) does
        # NOT use this stride — its DS pattern is masked-store with +=2 every
        # 4 cal_idx (golden L18 line 2275). The override table sets
        # storer_step=2 there; gate this 128 override on FSRCNN topology.
        if acc == 5 and not is_sd_unet:
            P.storer_step = 128

    # Phase 21: Pool-preceding conv detection. SD-UNet has explicit pool2d
    # layers (idx=3,6,9,12) preceded by conv layers (idx=2,5,8,11). The
    # preceding conv's DataStorer must emit is_pooling=1 with a height-
    # dependent pooling_out_mode so the hardware writes the pooled result
    # alongside the conv output. FSRCNN has no pool2d layers, so this loop
    # is a no-op there (has_pool_output stays False).
    for L, P in zip(layers, plans):
        next_layers = [M for M in layers if M.idx == L.idx + 1]
        if next_layers and next_layers[0].op == "pool2d":
            P.has_pool_output = True
            # pooling_out_mode depends on the spatial input height:
            #   h_in >= 128 → 0,  32 <= h_in < 128 → 1,  h_in < 32 → 2
            h = getattr(L, "h_in", 0) or 0
            P.pool_output_mode = 0 if h >= 128 else (1 if h >= 32 else 2)

    # Phase 30: SD-UNet per-layer DataLoader bas_addr per-row advance overrides.
    # The default emitter formula (2*(w_words-1) for padding rows, 2*w_words for
    # non-padding rows) is correct for layer 0 (DL reads from SRAM), but golden
    # sd_sr_codegen.py shows that for layers 1..18 the DL reads from upstream
    # buffers with a wide variety of stride conventions. Index = conv emission
    # order (the same key the emitter uses for DL.layer_idx). None preserves
    # the default formula. Gated on SD-UNet topology so FSRCNN is unaffected.
    if is_sd_unet:
        # (advance_padding_row, advance_non_padding_row) per conv-emission slot.
        unet_dl_advance = [
            None,         # 0: cin=1 cout=4 (initial SRAM read — formula 2/4 correct)
            (1, 2),       # 1
            (1, 2),       # 2
            (0, 4),       # 3
            (0, 8),       # 4
            (0, 4),       # 5
            (0, 8),       # 6
            (0, 4),       # 7
            (0, 8),       # 8
            (0, 8),       # 9
            (0, 4),       # 10
            (3, 4),       # 11
            (0, 2),       # 12
            (0, 8),       # 13
            (0, 8),       # 14
            (0, 8),       # 15
            (0, 8),       # 16
            (1, 2),       # 17
            (1, 2),       # 18
        ]
        # Right-half macro tile bas_addr_hint overrides (conv-emission order).
        # Default formula: hint += 288 per 128-wide tile (correct for emit 0,
        # SRAM input with 1-channel × 144-row = 2 words/row × 144 = 288 offset).
        # For subsequent layers the right-half start depends on the input buffer
        # layout from the preceding layer's output. None = keep default 288.
        #   emit 1 (cin=4):  right reads from 4ch×144 offset = 4×144 = 576
        #   emit 2 (cin=4):  both halves read the SAME input region → hint = 0
        #   emit 17 (cin=8, dual-region split=4):
        #       right reads at split×tile_h offset = 4×144 = 576 (same as emit1)
        #   emit 18 (cin=4): same-region layout as emit 2 → hint = 0
        unet_dl_right_hint = [
            None,  # 0: default 288 is correct for SRAM input
            576,   # 1
            0,     # 2
            None, None, None, None, None, None, None, None, None, None, None, None,
            None, None,  # 15, 16
            576,   # 17
            0,     # 18
        ]
        conv_emit_idx = 0
        for L, P in zip(layers, plans):
            if L.op not in ("conv2d", "deformable_conv2d", "offset_gen"):
                continue
            if conv_emit_idx < len(unet_dl_advance):
                adv = unet_dl_advance[conv_emit_idx]
                if adv is not None:
                    P.dl_advance_pad, P.dl_advance_nopad = adv
            if conv_emit_idx < len(unet_dl_right_hint):
                hint_override = unet_dl_right_hint[conv_emit_idx]
                if hint_override is not None and len(P.w_macro_tiles) == 2:
                    w0, w_sz, _ = P.w_macro_tiles[1]
                    P.w_macro_tiles = [P.w_macro_tiles[0], (w0, w_sz, hint_override)]
            conv_emit_idx += 1
    return plans
