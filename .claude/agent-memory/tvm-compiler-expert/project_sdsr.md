---
name: SDSR Compiler Project Context
description: SDSR CNN accelerator compiler frontend — architecture, current status, and key design decisions
type: project
---

TVM Relay-based compiler frontend for SDSR accelerator, targeting FSRCNN and UNet SR models.

**Current status (as of 2026-04-24):** Core pipeline complete, FSRCNN golden alignment at 1273/1274 instructions (0 functional differences).

**Four-stage pipeline:**
1. Frontend: relay.from_onnx / from_pytorch, InferType only, no fusion
2. LayerDesc extraction + fusion passes (OffsetGen, Activation)
3. Tiling: 5 named templates (A/B/C/D/E/F + deformable + offset_gen), all hand-designed
4. Emit + PostPass: EmitterState state machine, dependency analysis, virtual register allocation

**Key design decisions:**
- Relay IR used only as a structured parse tree (no TVM lowering, no TE schedules)
- LayerDesc is a flat dataclass — neutral IR between Relay and hardware ISA
- Tiling templates are hand-coded from golden reference (sd_sr_codegen.py)
- PostPass dependency analysis is ported verbatim from golden; deviations break correctness
- `src4 = src_code[2]` (not src_code[3]) is an intentional hardware quirk that must be preserved
- Global mutable state in isa.Inst (current_code_num, code_list) — reset before each emit_program call
- `id(expr)` must NOT be used for TVM node deduplication; use TVM's __hash__ instead
- feature_buf initialized to 'b' so layer-0 writes to 'a' (matching golden)
- QuantLoader uses 1-based layer_idx in golden format

**Why:** SDSR is a fixed-ISA accelerator; the compiler is a structured codegen tool, not a general optimizer. Golden parity is the primary correctness criterion.

**How to apply:** Suggestions should respect the "golden-first" constraint. Refactoring that could break golden parity needs careful validation. Performance optimization opportunities are secondary to correctness.
