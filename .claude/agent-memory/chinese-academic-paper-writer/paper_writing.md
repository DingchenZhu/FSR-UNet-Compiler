---
name: Paper Writing Context
description: Sections written, writing style, key arguments per chapter
type: project
---

**Sections written**: Chapter 2 (background) — docs/paper_background.md. Chapter 3 (system design), Chapter 4 (custom op support), Chapter 5 (optimization passes) — docs/paper_chapters_3_4_5.md, written 2026-04-22. Chapter 6 (experiments & evaluation) — docs/paper_chapter_6.md, written 2026-04-22. Chapter 1 (introduction) — docs/paper_chapter_1.md, written 2026-04-23. Chapter 7 (conclusion) — docs/paper_chapter_7.md, written 2026-04-23. Abstract (Chinese + English) — docs/paper_abstract.md, written 2026-04-23.

**Writing style**: Formal Chinese academic style, active voice, technical details grounded in source code. Sentences have rhythm. Technical terms in Chinese with English in parentheses on first use. Avoid translation-feel phrases.

**Chapter 3 key arguments**:
- TVM Relay IR chosen for its operator-level abstraction (not schedule-level), type system, and extensibility
- Dual frontend (from_onnx/from_pytorch) solves cross-framework compatibility
- _collect_calls_exec_order DAG traversal is the core IR extraction mechanism
- TVM ObjectRef id() instability was a P0 bug requiring deep TVM internals knowledge
- LayerDesc and TilingPlan as clean separation between frontend semantics and backend encoding

**Chapter 4 key arguments**:
- DeformableConv2d cannot use TVM generic lowering — hardware has dedicated OffsetLoader + bilinear WeightLoader units
- torchvision auto-conversion to nn.deformable_conv2d is a key insight that avoids custom op registration
- line_buffer_idx invariant is critical: DL and WL must share the same value, toggle only after both
- QuantLoader conv_layer_counter: only conv/dconv/offset_gen layers count, not prelu/pool

**Chapter 5 key arguments**:
- OffsetGenerator fusion reduces layer count from 23→19 and instruction count from 864→840
- The fusion is semantically necessary: without it, pool2d→PseudoOp (wrong), conv2d→standard path (wrong dest)
- Post-pass dependency analysis: 7 rules covering all producer-consumer relationships across 7 ISA types
- src4 quirk must be preserved verbatim — it is not a bug to fix
- load_next enables multi-frame pipeline overlap: preamble[0-4] → image[5] → layer-0 → load_next → inter_layer

**Chapter 5 new sections added 2026-04-27** (§5.5–§5.7 in paper_chapters_3_4_5.md):
- §5.5: SD-UNet full-height streaming scheduling — tile_h=None vs tile_h=32; single code path controlled by PipelineConfig.tile_h; AveragePool halving via TVM shape inference (no manual logic needed)
- §5.6: pool-while-store transparency — no independent AveragePool instruction in SDSR ISA; pool2d kept in LayerDesc (for shape info), emitted as zero instructions (pass); P1 pending: inject is_pooling=1 into preceding conv's DataStorer
- §5.7: TilingPlan parameter calibration — 17 layers × 12 params; shape-keyed _UNET_LAYER_TABLE + idx-keyed _UNET_IDX_OVERRIDE_TABLE for disambiguation; oc_inner outer loop (L14/L16, ds_oc_stride); result: 10487→17079 instr (×1.64→×0.996), FSRCNN 0 regression
- Chapter 5 OLD §5.4 summary table (5-3) was in the old position, now it precedes the new §5.5 section

**Chapter 6 key arguments** (updated 2026-04-27):
- Core result: 1273/1273 (load_next=False) and 1274/1274 (load_next=True) exact instruction count match
  - QL=12, DL=524, WL=524, DS=116, OL=96, ODS=1; ODL=0 or 1 depending on load_next mode
- emit_image_load parameter: allows aligning with standalone sr_inst() which assumes image pre-loaded by UNet
- ping-pong buffer fix: EmitterState.feature_buf init='b' → L0 DS writes 'a', then a→b→a alternating; offset_gen layers do NOT flip feature_buf; last layer DS writes fsrcnn_output_buffer
- acc_mode/store_mode auto-derivation: _derive_acc_store_mode() in tiling.py; rules cover all 7 layer-type/activation combinations; pool-while-store (4,3) triggered by conv+prelu before offset_gen layer
- 8/12 FSRCNN layers match immediately (4 offset_gen + 4 deformable_conv layers)
- 4/12 layers required Tiling template fixes (Template C for cin=1 k=3, D/E for 1×1, F for last-part)
- Field-level diff analysis: 1159/1274 instructions have at least one field difference; two categories:
  1. External-input fields (bas_addr=831, quant_mode=8): not compiler-derivable; analogous to link addresses
  2. ISA template params (line_buffer_reshape=512, line_buffer_row_shift=320, etc.): refinement work, not architectural errors
- Structural correctness established: instruction type sequence, buffer direction, activation fusion, tiling structure all correct
- quant_mode still requires external calibration table; acc_mode/store_mode now fully auto-derived
- SD-UNet validation FULLY COMPLETED (Phase 11-32): USR_Net_109_nopad.onnx confirmed as sd_inst model; FINAL RESULT: 17155/17155 instructions, 0 functional diff ✅
- Phase 20: final 76-instruction gap closed (conv11 ds_last_transfer_num=0 signal, oc_inner adjustment, DepthToSpace field injection)
- Phase 21: is_pooling=1 injection into pool-preceding conv DataStorer implemented (not just "P1 pending" anymore)
- Phase 32: 14,664 remaining field diffs systematically confirmed as NON-FUNCTIONAL via layer_diff.py + multiset analysis:
  - ~93% = WL is_new sequential vs interleaved scheduling (no computation impact)
  - ~0.4% = QL quant_reg_load_idx register slot choice (hardware symmetric dual-slot)  
  - ~0.7% = WL ordering artifacts at L=11 (bas_addr multiset identical, 480 WL instructions)
- §6.5.3 now: complete SD-UNet validation section with network structure table, phase-by-phase progress table, per-type count match table, and 14664-diff non-functional classification table
- §6.6 updated: both networks at 0 functional diff, 17155/17155 for SD-UNet confirmed
- load_next hoisting: theoretical optimization, ~30-line change, needs hardware simulation to quantify

**Chapter 1 key arguments**:
- Opens with SR/CNN hardware deployment gap as motivation
- Problem statement: sd_sr_codegen.py (~3800 lines, hardcoded, unmaintainable) as concrete symptom
- Three-axis framing of the problem: maintainability, extensibility, non-standard op coverage
- Related work gap: TVM/Ansor/Halide are GPU-oriented; none address the complete chain from QAT model to strict-ISA custom hardware
- Four contributions: (1) dual frontend + TVM hash bug fix, (2) OffsetGenerator fusion pass, (3) 4-level IR hierarchy + hardware constraint modeling, (4) 1273/1274 exact match validation
- Chapter organization paragraph covers Ch.2-7 concisely

**Chapter 7 key arguments** (MAJOR UPDATE 2026-04-30; Phase 34 update 2026-05-09):
- SD-UNet: 17155/17155, 0 functional diff, datapath equivalence FORMALLY VERIFIED via equivalence_check.py
- FSRCNN: 1273/1273 instruction count aligned. Phase 34 found FSRCNN has datapath divergence (not exposed before); conclusions in paper now only state "指令数对齐" for FSRCNN, NOT "0功能性diff". NEVER write "FSRCNN 0功能性diff" — this was disproven by Phase 34.
- Three-level contribution summary: IR layer, tiling/scheduling, hardware interface correctness
- Limitations now updated: (1) sequential scheduling only (not interleaved — source of 14664 non-functional diffs); (2) quant_mode external; (3) bas_addr partially derived (P0 done, P1 needs ILP); (4) ISA template params
- Future work EXPANDED: interleaved scheduling (P2), quant integration, ILP memory layout, load_next hoisting, MLIR dialect evolution, broader op coverage
- §7.5 closing: reframes to "SD-UNet datapath equivalence formally verified; FSRCNN instruction count aligned; both ready for hardware board-level validation"

**Tutorial for owner**: docs/tutorial_for_owner.md — written 2026-04-24, colloquial Chinese style (not academic), covers hardware, all 4 pipeline stages, 6 key bugs with root-cause stories, validation results, and 8 FAQ answers for tomorrow's demo. Target reader: project PI who knows the big picture but not code details.

**How to apply**: Maintain consistency with these arguments when extending or revising the paper. Do not contradict findings established here.
