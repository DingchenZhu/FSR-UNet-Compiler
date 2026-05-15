---
name: SD-UNet TilingPlan calibration (Phases 15 + 18 + 20 + 22 + 23 + 24 + 26 + 27 + 28 + 29 + 30)
description: SD-UNet override-table approach across tiling.py and addr_alloc.py — count match (17155) plus skip-region anchors, decoder output overrides, storer_step calibration, pool-tracker bas_addr derivation, mask-store + dual-region DL + per-group bases + per-layer DL advance + QL-reuse-across-macros
type: project
---

SD-UNet total reached **17155 / 17155 (exact match)** after Phase 20 P1 fixes. All op_code counts (QL=37, DL=7824, WL=7824, DS=1468) match golden exactly. FSRCNN unaffected (1273 sr_inst / 1274 pipeline default, 0 field-filtered diffs).

**Why:** Phase 14 had 10487. Phase 15 reached 17079 (-76). Phase 18 added DepthToSpace fields (no count change). Phase 20 closed the residual -76 by adding two QL dispatch modes and fixing idx=15/16 cin_group / load_total / h_out_per_step.

**How to apply:**
- Calibration lives in `tiling.py::_UNET_LAYER_TABLE` (shape-keyed) and `_UNET_IDX_OVERRIDE_TABLE` (idx-keyed for shape collisions like encoder conv6 vs decoder conv12 both being (18,32,16,64,3,2)).
- Activated only when `tile_h is None` (full-height streaming mode) — FSRCNN tiled-32 path is bypassed.
- TilingPlan fields: `oc_inner`, `ds_oc_stride`, `ic_only_no_ky`, `is_pixelshuffle` + 5 pixelshuffle field overrides (Phase 18), and `ql_per_macro` / `ql_per_oc_iter` (Phase 20).
- `choose_tiling()`: when override dict contains `load_total_num`, do NOT recompute from `effective_tile_h // h_out_per_step` (idx=15 needs explicit 5 = ceil(18/4), floor would be 4).
- Emitter dispatch:
  - Default: 1 QL before all macro tiles.
  - `ql_per_macro=True`: 1 QL per macro tile, both share quant_config_idx.
  - `ql_per_oc_iter=True`: QL emitted inside oc_inner loop; quant_config_idx toggled BETWEEN oc iterations so each QL/DS pair sees a consistent value. End-of-layer toggle still applies (oc=2 means net 3 toggles ≡ 1 mod 2 — invariant preserved).

**Phase 20 layer-table changes (committed values):**

| layer.idx | shape | h_out_step | cin_group | flags |
|---|---|---|---|---|
| 1, 2 | (144,256,4,4,3,1) | 2 | 4 | ql_per_macro=True |
| 15 | (18,32,128,16,3,2) | 4 (was 1) | 16 (was 1) | load_total_num=5 explicit |
| 16 | idx-override | 1 | 1 (was 4) | (golden L12 ic=1, ky-only) |
| 17 | (36,64,32,16,3,1) | 1 | 4 (was 2) | oc_inner removed |
| 18 | (36,64,16,32,3,1) | 1 | 2 (was 4) | oc_inner=2, ds_oc_stride=8, ql_per_oc_iter=True |
| 19 | (72,128,16,8,3,1) | 1 | 4 (was 2) | oc_inner removed |
| 20 | (72,128,8,16,3,1) | 1 | 2 (was 4) | oc_inner=2, ds_oc_stride=1, ql_per_oc_iter=True |
| 21 | (144,256,8,4,3,1) | 2 | 8 | ql_per_macro=True |
| 22 | (144,256,4,1,3,1) | 2 | 4 | ql_per_macro=True |

**Phase 18 pre-DepthToSpace conv field map (still load-bearing):**

| our layer.idx | shape (h,w,cin,cout,k,g) | golden QL.layer_idx | pix_out_mode | acc_mode | store_mode | transfer_num | stride |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | (9,16,64,256,3,8) | 11 | 0 | 3 | 0 | 0 | 18 |
| 16 | (18,32,16,64,3,2) | 13 | 1 | 1 | 3 | 1 | 8 |
| 18 | (36,64,16,32,3,1) | 15 | 2 | 1 | 1 | 1 | 0 |
| 20 | (72,128,8,16,3,1) | 17 | 2 | 6 | 2 | 1 | 144 |

**Phase 22 + 23 — pool address & skip-aware DataLoader wiring (2026-04-28):**

Pool-preceding conv layers (idx=2/5/8/11) now emit `base_addr_pooling` derived from per-layer override fields in `_UNET_IDX_OVERRIDE_TABLE`. Five fields drive the formula:

```python
pool_addr_start         # first DS base
pool_addr_stride        # increment per period
pool_addr_inc_period    # how many DS share same base (1 or 2)
pool_addr_macro_stride  # offset per macro W tile (idx=2 right half = +2)
pool_addr_group_stride  # offset per group_level1 (idx=11 conv7 = +72)
```

Final formula (shared by `_emit_w_macro_tile` and `_emit_group_w_tile`):
```
base = start + l1*group_stride + macro_idx*macro_stride + (load_idx//inc_period)*stride
```

Calibrated values (verified against `sd_sr_codegen.py` line numbers):
- idx=2 conv1_2: start=1152 stride=4 period=1 macro_stride=2
- idx=5 conv3:   start=1728 stride=4 period=2
- idx=8 conv5:   start=2016 stride=4 period=2
- idx=11 conv7:  start=2016 stride=8 period=2 group_stride=72

**Skip-aware DataLoader** (`emit_layer`): when `layer.skip_sources` non-empty, use `addr_map[min(skip_sources)]` instead of `addr_map[last_feature_layer_idx]` as the input base. SD-UNet idx=15/17/19/21 are detected as skip consumers; their DL bas_addrs now derive from encoder addresses.

**Phase 24 — DepthToSpace transparent + fusion-pass index remap (2026-04-28):**
- Added `nn.depth_to_space`, `nn.space_to_depth`, `sigmoid` to `_KNOWN_HARMLESS_OPS` (auto-included in `_STRIP_THROUGH_OPS`) so `_strip_to_data_call` traces through pixel-shuffle to the upstream conv.
- Added `_remap_skip_sources()` helper in `ir/fusion_pass.py`. Both `fuse_offset_generators` and `fuse_activations` now build an `old_to_new: Dict[int, int]` (with merged old indices for collapsed pairs both mapping to the same new position) and remap `LayerDesc.skip_sources` before re-numbering. Required because `extract_layer_descs` populates skip_sources in the 41-layer pre-fusion index space; without the remap, post-fusion 23-layer skip_sources held stale/invalid indices.
- Verified post-fusion skip_sources for SD-UNet: L15=[14,11], L17=[16,8], L19=[18,5], L21=[20,2] — channel sums and H/W match U-Net topology (encoder + DepthToSpace upstream).
- Instruction counts unchanged: SD-UNet 17155, FSRCNN 1273.

**Phase 26 — Decoder output base override (2026-04-29):**
- Added `_SD_UNET_DECODER_OUTPUT_BASE_BY_IDX = {14: 2016, 16: 2016}` to `ir/addr_alloc.py`. These are the bottleneck conv10 (idx=14) and first decoder pre-DTS conv12 (idx=16) which write into the post-consumption skip region [2016, ...] (matches `sd_sr_codegen.py` lines 1432, 1623). Linear-scan would otherwise place them at 0 and 2160 respectively.
- Added `_is_sd_unet_topology(layers)` (any `skip_sources` non-empty) to gate the decoder override — FSRCNN/sequential models are unaffected.
- `_build_skip_region_table()` now returns BOTH skip producer addresses and decoder output overrides; the merged dict `addr_map.update(skip_table)` semantics in `allocate_addresses()` are unchanged.
- Verified: idx=14 first DS = 2016 (was 0), idx=16 first DS = 2016 (was 2160). Counts unchanged (17155 / 1273). Other decoder pre-DTS layers (idx=18→1728, idx=20→1152) already match via natural linear-scan conflict resolution against c3/c5 producer live ranges — NOT in override table.

**Phase 27 — storer_step + bas_addr derivation + idx=13 base (2026-04-28):**
- Updated `_UNET_LAYER_TABLE` storer_step values per-shape to match golden:
  conv2/3/4/5/6/13/15 step=8; conv7 (h-cont.) step=1; conv8 step=4; conv10
  step=2; conv11 step=8; conv12 (idx-keyed override) step=16; conv14 step=16;
  conv16 step=2. Verified DS base_addrs_res sequences match golden head/tail
  for all SD-UNet conv layers except idx=22 (which uses a conditional mask-
  store pattern not expressible as a single integer step).
- Gated `acc==5 → storer_step=128` on `is_sd_unet = any(L.skip_sources)` to
  protect SD-UNet's terminal idx=22 (acc=5 last-no-act) override (=2). FSRCNN
  last_part still gets 128 since its layers carry no skip_sources.
- Added `13: 36` to `_SD_UNET_DECODER_OUTPUT_BASE_BY_IDX` so conv8's first
  DS writes to base 36 (golden line 1333; the [0,36) prefix is reserved for
  conv11 g_idx=1's interleaved scheduling).
- Added `EmitterState.last_feature_pool_addr_start` and reworked
  `emit_layer`'s input-address derivation as a 3-way switch:
  * skip_sources non-empty → pick source with **lowest addr_map value** (was:
    lowest idx). Crucial for idx=15 (skip=[14,11], addr_map[14]=2016 <
    addr_map[11]=2160, golden expects 2016).
  * non-skip following a `has_pool_output=True` layer → use that layer's
    `pool_addr_start` (1152/1728/2016 for idx=2/5/8/11 → consumed by
    idx=4/7/10/13). Resets to -1 after non-pool layers.
  * sequential default → `addr_map[last_feature_layer_idx]`.
- Counts unchanged (17155 / 1273); FSRCNN field-filtered diff still 0.

**Phase 28 — close 4 Phase 27 P2 gaps (2026-04-28):**
- **Fix 3 — conv12 (idx=16) g=2 strides**: added `dl_level2_stride=36,
  ds_level2_stride=4` to `_UNET_IDX_OVERRIDE_TABLE[16]`. Required swapping
  `choose_tiling()` order: `_apply_group_params` runs FIRST, then SD-UNet
  override (so per-idx values can override conv6/L7 defaults). idx=10
  (encoder conv6) unaffected — its override has no level2_stride entries.
- **Fix 4 — conv11 (idx=15) per-group base**: added per-group base detection
  in `_emit_group_conv()`. When `len(skip_srcs) == group_level2 and
  group_level1 == 1 and group_level2 > 1` and skip sources have distinct
  addr_map entries, build `per_group_dl_bases = [addr_map[skip_srcs[i]]]`.
  Each l2 iteration uses `per_group_dl_bases[l2]` instead of
  `layer_input_bas_addr + l2*stride`. For idx=15 → group 0 reads 2016
  (addr_map[14]), group 1 reads 2160 (addr_map[11]). FSRCNN unaffected
  (no skip_sources, no group conv).
- **Fix 1 — idx=22 mask-store**: added 3 TilingPlan fields (`is_mask`,
  `storer_increment_period`, `mask_macro_offset`). idx=22 override
  table sets `is_mask=True, storer_increment_period=4,
  mask_macro_offset=1`. Emitter `_emit_w_macro_tile`:
  - DS field selection: `is_mask` branch BEFORE `is_pixshuffle_legacy`
    sets `acc_mode=0, transfer_num=0, store_mode=1, stride=0,
    pix_out_mode=0, is_pixelshuffle=0`.
  - `is_mask=is_mask_field, is_new = (load_idx % period == 0) ? 1 : 0`.
  - storer_bas_addr increments by `storer_step` only when
    `load_idx % period == period - 1`.
  - `tile_half_offset` for right macro = `mask_macro_offset` (=1) instead
    of `tile_h * 4` when `is_mask=True`.
- **Fix 2 — idx=21 dual-region DataLoader**: added `dual_split,
  dual_region_jump` computation at `_emit_w_macro_tile` head. Detection:
  `len(skip_srcs)==2 and group_count==1 and cin_group%2==0 and
  cin_group>1 and region_jump>0`. cin_g >= split adds `dual_region_jump`
  to bas_addr. For idx=21 with skip=[20,2], addrs [1152,0], split=4,
  jump=576: ic 0..3 reads at 0/144/288/432, ic 4..7 reads at
  1152/1296/1440/1584 (matches golden L17 line 2329).

**Phase 29 — DS field overrides, layer_idx canonicalization, ql_per_macro
correction (2026-04-29):**
- Added `ds_stride: Optional[int]` to TilingPlan. Default branch of
  emitter's DS field selection now reads `plan.ds_stride if not None else
  plan.tile_h`, so per-shape overrides can pin stride to 0 for SD-UNet
  ky-software-loop convs without affecting FSRCNN Template A/B.
- Added `acc_mode`/`store_mode`/`ds_stride` overrides to 7 SD-UNet shape
  entries in `_UNET_LAYER_TABLE`: (72,128,4,8) (72,128,8,8) (36,64,8,16)
  (36,64,16,16) (18,32,16,64,g=2) (18,32,64,64,g=8) (36,64,32,16)
  (72,128,16,8). Most use (1,1,0); conv7 g=8 (18,32,64,64,8) uses (1,2,18).
- `plan_all()` re-applies these overrides AFTER `_derive_acc_store_mode`,
  but ONLY when SD-UNet topology is detected (`any(L.skip_sources)`).
  FSRCNN flow unaffected.
- Added `EmitterState.dl_layer_idx` and `ql_layer_idx` properties using
  `conv_layer_counter`. DL/QL emit calls now pass these instead of
  `layer.idx`, eliminating the gap (4→3, 5→4, 7→5, …) between our
  post-fusion idx and golden's contiguous 0-based conv index. QL uses
  `dl_layer_idx` (matches fresh sd_inst dump where QL.layer_idx ==
  DL.layer_idx); the archived golden `pseudo_code_load_next_mid.txt`
  uses 1-based QL (off by 1) — that's a known divergence between the
  archived file and current sd_sr_codegen output.
- Removed `ql_per_macro: True` from (144,256,4,4,3,1). Golden conv1_1 /
  conv1_2 emit ONE QL per layer (sd_sr_codegen line 306-313 / 471-478),
  NOT one per macro tile. The flag was a calibration miss carried over
  from the archived golden file; fresh dump expects 1 QL.
- Final SD-UNet count: 17153 instructions (was 17155 → -2 from removing
  redundant L=1/L=2 QLs). Fresh `sd_inst` dump = 17154 (the +1 is the
  end-of-stream OffchipDataLoader for next-image preload, gated by
  `load_next` — separate code path).
- FSRCNN regression: 1274 lines (1273 sr_inst + 1 pipeline-default
  OffchipDataLoader); field-filtered diff vs sr_inst golden = 0.

**Phase 29 diff metrics (vs fresh sd_inst dump, full skip_set):**
- Total field-level diffs: 32630 → 27375 (−5255, ≈ 16%).
- All zero-count alignments now match except: L=5/L=7 WL bas_addr (manager
  state, in project skip set), L=17 DS base_addrs_res off-by-1 (1 vs 2),
  L=18 DL bas_addr off-by-1 (1 vs 2 mask layer right macro).
- Project-skip-set (bas_addr/base_addr_pooling/quant_config_idx/etc
  skipped): 4281 diffs total. Largest contributors: L=11/L=13/L=15/L=17
  WL bas_addr (manager state, multi-thousand entries each).

**Per-layer diff helper script:** `tools/layer_diff.py`. Buckets DL/QL/WL/DS
into 19 logical conv layers by sorted distinct DL.layer_idx (handles ours'
gapped numbering and golden's interleaved scheduling uniformly). QL
forward-binds to next DL; WL/DS/Offchip* backward-bind. Multiset diff per
layer with field-level breakdown (`--mode details`) plus zero-address
stats (`--mode zero`). Run vs `/tmp/sd_inst_dump.txt` (fresh) or
`golden/pseudo_code_load_next_mid.txt` (archived).

**Phase 30 — DL per-row advance override + QL reuse across macros (2026-04-29):**
- Added `dl_advance_pad: Optional[int]`, `dl_advance_nopad: Optional[int]`
  to `TilingPlan`. Default None preserves the historic `2*(w_words-1)` /
  `2*w_words` formula (FSRCNN unaffected, layer 0 SRAM-read unaffected).
- Emitter checks the override at THREE sites (standard conv path,
  group conv path, deformable conv path): when the field is set, use the
  per-layer pair literally; otherwise fall through to `w_words` formula.
- `tiling.plan_all()` post-processes `is_sd_unet=True` plans with a
  19-entry `unet_dl_advance` table indexed by conv-emission order
  (matches `EmitterState.dl_layer_idx`). Index 0 → None (keep formula
  for the SRAM-read first conv). Values lifted from `sd_sr_codegen.py`:
  L1/L2/L17/L18 → (1,2); L11 → (3,4); L3/L5/L7/L10/L12 → (0,4) or (0,2);
  L4/L6/L8/L9/L13..L16 → (0,8).
- QL bas_addr fix in `emit_layer`: for `ql_per_macro=True`, undo the
  prior `quant_bas_addr += transnum` advance before emitting the QL of
  macro_idx > 0 so all macro tiles in a layer share the same QL bas_addr
  (matches golden — QL is not retiled along W).
- Verified: SD-UNet count unchanged at 17155. Total field-level diffs vs
  archived golden dropped 31188 → 22850 (−8338, ≈ 27%). Within-macro
  per-row DL advance now matches exactly (e.g. L=1 first 288 entries
  identical between ours and golden); the residual L=1 DL diffs after
  index 288 are macro-tile base offset issues (separate from advance).
  FSRCNN field-filtered diff = 0 (1273 lines).

**Remaining work (P3):**
- **Macro W tile bas_addr_hint per-layer** — `_macro_w_tiles` uses a
  global `+288` step for w=128; gold expects per-layer values (e.g.
  L=1 right-half base = 576 vs ours 288). Add `dl_macro_hint`
  override or move bas_addr_hint into `_UNET_LAYER_TABLE`.
- **idx=17/19 dual-region cin step ≠ tile_h** — Phase 28 Fix 2 assumes
  within-region cin step is `tile_h`, but golden L13/L15 use step=4
  (not tile_h=36/72). Current emitter generates step=tile_h for these
  layers, differs from golden but doesn't change instruction count.
- **WL bas_addr divergence (L=5/L=7 zero count, L=11+ thousands of values)**:
  WeightLoaderManager.bas_addr_cur state is not tracked; golden has
  is_skip resets that bring some WLs back to 0. To eliminate, port
  `weightloadermanager.bas_addr_cur` semantics fully (per-layer reset +
  is_skip-driven non-advance).
- **DL.bas_addr off-by-N**: many layers have DL bas_addr ours < golden by
  small amounts (e.g. L=3: 1154 vs 1160 → -6). Caused by initial
  `data_bas_addr` offset for ky-software-loop layers (golden `data_bas_addr
  + 4*load_per_cal_index - {4 if cal_idx<padding and load_per_cal_index
  in (1,2) else 0}`); not yet replicated.
- **L=11 idx=15 group-conv field cluster** (transnum/wl_lrs/parall_mode/
  reshape): 1944 diffs. Group conv path has different field defaults
  vs golden — needs alignment.
- Real concat data path / end-to-end simulation still pending.
