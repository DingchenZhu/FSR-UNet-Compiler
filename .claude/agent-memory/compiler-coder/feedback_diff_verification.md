---
name: FSRCNN golden "0 diff" means field-filtered, not raw
description: How to interpret FSRCNN golden regression — verification uses field-filtered comparison (skips address/post-pass fields)
type: feedback
---

"FSRCNN golden 0 diff" verification compares only semantic/template fields; address-like fields are expected to differ and must be skipped.

**Why:** The compiler does not track concrete weight/quant/DataStorer addresses the way the hand-written golden (`sd_sr_codegen.py`) does. Memory manager state (`weightloadermanager.bas_addr_cur`, `datastorermanager.base_addr_pooling_cur`, `quant_config_idx` swap, post-pass dependency/dest/src1-4/is_offset) is not derivable from pure topology. These have always been skipped, per `compiler_progress.md` under "已跳过字段".

**Skip set for golden diff:**
`bas_addr`, `base_addr_pooling`, `quant_config_idx`, `dependency`, `dest`, `src1`, `src2`, `src3`, `src4`, `is_offset`. Note that `base_addrs_res` is NOT in the skip set — it IS a semantic correctness check (DS base_addrs_res stride per iteration is template-specific and must match golden).

**How to apply:** When verifying FSRCNN golden regressions, use a field-filtered diff (output fsrcnn_test/pseudo_instructions.txt vs sr_inst_golden.txt). Raw `pipeline.py --golden` returns nonzero because it includes skip-set fields. To reproduce 1273/1274 golden lengths, run with `emit_image_load=False, load_next=False/True, is_first=False` (CLI does not expose these flags — invoke `run_pipeline()` via Python). Expected totals: 1273 (load_next=False), 1274 (load_next=True).
