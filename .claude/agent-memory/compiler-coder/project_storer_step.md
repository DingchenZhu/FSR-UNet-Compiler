---
name: Per-template DataStorer storer_step
description: TilingPlan.storer_step carries per-template base_addrs_res stride; replaces emitter's old hardcoded +=2
type: project
---

`TilingPlan.storer_step` is the per-iteration DataStorer `base_addrs_res` increment. Formerly hardcoded as `+=2` in `emitter._emit_w_macro_tile` and `+=4` in `_emit_deformable_conv`; now driven by the plan.

**Why:** The hardcode matched only Template A/B. Template C (cin=1,k=3) and Template D (k=1,cin≤8) need +=1; Template E (k=1,cin>8) needs +=4; deformable needs +=4; FSRCNN pixelshuffle needs +=128. Wrong stride corrupts DataStorer output layout and shows up as base_addrs_res mismatches in the golden diff.

**How to apply:** When adding a new template in `tiling/tiling.py::choose_tiling`, set `storer_step` explicitly in the branch. `plan_all()` overrides it to 128 when `acc_mode=5` (pixelshuffle terminal layer). Both `emitter._emit_w_macro_tile` and `_emit_deformable_conv` now read `plan.storer_step`.

**Golden-verified values** (from `sd_sr_codegen.py`):
- Template A/B (h_out_per_step=2): 2
- Template C (cin=1, k=3): 1
- Template D (k=1, cin≤8): 1
- Template E (k=1, cin>8, FSRCNN L1): 4
- Template F (k=3, cin>8, cout≤8, non-pixshuf): 4 (default; FSRCNN L11 overridden to 128 via plan_all pixshuf branch)
- deformable: 4
- pixelshuffle (acc_mode=5): 128 (FSRCNN last_part, fsrcnn_output_buffer stores 4 values per row)

Unknown / untested for UNet: UNet Template E/F variants may differ (e.g. some UNet 3×3 layers use step=8). Revisit when tackling UNet golden parity.
