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

**Chapter 6 key arguments** (updated 2026-04-23):
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
- UNet full validation pending model identity confirmation (USR-Net.onnx ≠ sd_inst 19-layer UNet)
- load_next hoisting: theoretical optimization, ~30-line change, needs hardware simulation to quantify

**Chapter 1 key arguments**:
- Opens with SR/CNN hardware deployment gap as motivation
- Problem statement: sd_sr_codegen.py (~3800 lines, hardcoded, unmaintainable) as concrete symptom
- Three-axis framing of the problem: maintainability, extensibility, non-standard op coverage
- Related work gap: TVM/Ansor/Halide are GPU-oriented; none address the complete chain from QAT model to strict-ISA custom hardware
- Four contributions: (1) dual frontend + TVM hash bug fix, (2) OffsetGenerator fusion pass, (3) 4-level IR hierarchy + hardware constraint modeling, (4) 1273/1274 exact match validation
- Chapter organization paragraph covers Ch.2-7 concisely

**Chapter 7 key arguments**:
- Summary reviews all four contributions tied to concrete outcomes
- Limitations: three explicit gaps: quant_mode (external calibration), bas_addr (external memory layout), UNet validation pending
- Also flags ISA template params (line_buffer_reshape etc.) as refinement work
- Future work: quant calibration integration, memory layout modeling, op coverage expansion, UNet alignment, load_next hoisting
- Closing sentence frames the system as an "instruction-correct frontend prototype" on path toward full-stack production compiler

**How to apply**: Maintain consistency with these arguments when extending or revising the paper. Do not contradict findings established here.
