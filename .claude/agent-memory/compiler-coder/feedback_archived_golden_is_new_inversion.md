---
name: archived golden is_new convention is inverted vs sd_sr_codegen.py
description: pseudo_code_load_next_mid.txt flips WL is_new bit; do not globally invert per user instruction
type: feedback
---

The archived golden file `golden/pseudo_code_load_next_mid.txt` consistently uses the COMPLEMENT of `sd_sr_codegen.py`'s is_new convention for `WeightLoader.is_new`. Across SD-UNet conv layers L=0..18, every WL emits `is_new=1` where the codegen says `is_new=0` and vice versa.

**Why:** The archived golden was generated from an earlier version of `sd_sr_codegen.py` whose `is_new` semantic was flipped from the current source (the codegen comments still say `is_new = 0 if ... else 1`, our compiler emits this same convention, but the archive emits `is_new = 1 if ... else 0`). The bit's hardware meaning is unchanged — `is_new=1` = "start a new accumulator", `is_new=0` = "accumulate into the previous one" — only the polarity convention diverges.

**How to apply:** When the user asks to fix WL `is_new` diffs vs the archived golden, **do NOT** add a global `is_new = 1 - is_new` inverter. The user explicitly classified L=1 WL is_new (576 entries) as "调度差异，暂不修" (scheduling difference, don't fix). Treat is_new diffs as a known structural divergence between the two convention versions, present in roughly 4000 entries across L=0..18 in the archived golden. Field-filter with `is_new` in the skip set if you need a clean signal-to-noise diff for other fields. Confirmed in Phase 31 work (April 2026): leaving these alone preserved the user's intent and dropped non-is_new diffs significantly.
