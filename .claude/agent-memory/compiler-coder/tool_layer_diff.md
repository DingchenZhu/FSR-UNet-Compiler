---
name: tools/layer_diff.py per-layer field-level diff
description: How to use the per-layer multiset diff tool against fresh sd_inst dump or archived golden
type: reference
---

`tools/layer_diff.py` is the canonical per-layer field comparison tool for
SD-UNet golden parity work.

**Why:** The repo had no automated way to attribute diffs to specific conv
layers. Raw-line diff is useless because golden uses interleaved scheduling
while we emit sequential — same instruction set, different order.
layer_diff.py buckets into 19 logical conv layers and uses Counter-based
multiset comparison, so order-of-emission doesn't matter.

**How to apply:**
- Default golden: `/home/hansz/scratch-data/design/tvm-tiling/golden/pseudo_code_load_next_mid.txt`
  (archived; uses 1-based QL.layer_idx, has obsolete +=576 right-half DS).
- Fresh dump: regenerate via `scripts/dump_sd_inst_pseudo_code.py -o /tmp/sd_inst_dump.txt`
  in tvm-tiling. Has 17154 lines (vs ours 17153). Use `--golden /tmp/sd_inst_dump.txt`
  for accurate sd_sr_codegen comparison.
- Modes:
  - `--mode zero`: zero-count stats per (layer, op×address-field) pair. Best
    for diagnosing missing macro-tile resets, off-by-one base addrs.
  - `--mode summary`: total diff count per layer. Quickly identifies the
    biggest-impact layers.
  - `--mode details`: full field-level breakdown per layer, grouped by
    (op, set-of-differing-fields). Shows ours-value vs golden-value for one
    representative pair per group, plus the count.
- `--only-layer N`: focus output on a single logical conv layer (0..18).
- `--max-examples N`: cap pairwise pairs walked. Default unlimited.

**Skip set in tool** (script's `SKIP_FIELDS`): code_num, dependency, dest,
src1-4, is_offset, quant_config_idx. Address fields (bas_addr, base_addrs_res,
base_addr_pooling) are KEPT — they are semantic correctness checks.

**Layer bucketing algorithm:** sort the distinct DL.layer_idx values found
in the file; map each to logical 0..N-1. WL/DS/Offchip* backward-bind to
the most recent DL's logical layer. QL forward-binds to the NEXT DL's
logical layer. This handles both ours' gapped layer_idx (post-fusion 0,1,2,
4,5,7,8,…,22) and golden's interleaved emission (which would break a
naive first-encountered-claims-slot approach because golden emits
layer_idx=11 before 9/10).
