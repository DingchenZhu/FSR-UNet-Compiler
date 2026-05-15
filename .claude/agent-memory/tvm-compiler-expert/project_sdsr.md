---
name: SDSR Compiler Project Context
description: SDSR CNN accelerator compiler frontend — architecture, current status, and key design decisions
type: project
---

TVM Relay-based compiler frontend for SDSR accelerator, targeting FSRCNN and UNet SR models.

**Current status (as of 2026-04-28):** Instruction count fully aligned (17155/17155 UNet + FSRCNN). All functional fields correct. Address fields (base_addr_pooling, base_addrs_res start, DL bas_addr for skip) still placeholder=0. This is the primary gap before hardware deployment.

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

**Address allocation gap (researched 2026-04-28):**
- `ir/addr_alloc.py` _output_size_words() overcounts by cout factor (uses w*cout/64, should be w/64)
- `base_addr_pooling` is hardcoded 0 in emitter.py at L323/532/604; needs new pool_addr_map
- Buffer A is UNet's persistent skip-tensor store; buffer B is transient per-layer output
- Pool output starts immediately after the main skip tensor in buffer A (cumulative layout)
- base_addr_pooling start values: layer2=1152, layer4=1728, layer6=2016, layer8=2016
- Recommended plan: Phase A (1-2 days) hardcode golden values in TilingPlan overrides; Phase B (3-5 days) generalize with TilingPlan-driven static layout pass
- Research document: docs/addr_alloc_research.md

**Why:** SDSR is a fixed-ISA accelerator; the compiler is a structured codegen tool, not a general optimizer. Golden parity is the primary correctness criterion.

**How to apply:** Suggestions should respect the "golden-first" constraint. Refactoring that could break golden parity needs careful validation. Performance optimization opportunities are secondary to correctness.
