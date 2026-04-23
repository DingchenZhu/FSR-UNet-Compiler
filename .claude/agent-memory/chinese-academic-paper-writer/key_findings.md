---
name: Key Technical Findings
description: Non-obvious bugs discovered and fixed during implementation, plus critical invariants
type: project
---

**1. TVM ObjectRef id() instability (CRITICAL BUG)**
- Problem: _collect_calls_exec_order used id(expr) to track visited nodes. TVM ObjectRef creates a new Python wrapper on each attribute access, so id() returns a different value even for the same underlying C++ node. This caused exponential re-traversal of shared subgraphs → timeout.
- Fix: Use `expr in visited` (set membership) instead of `id(expr) in visited`. TVM's __hash__ is based on the C++ object pointer (stable), and __eq__ uses same_as() (consistent with __hash__).
- Impact: Compilation time went from timeout → 0.016s.
- Code location: ir/layer_desc.py, _collect_calls_exec_order(), line 85-87

**2. line_buffer_idx invariant (CRITICAL for golden parity)**
- Rule: DataLoader and WeightLoader MUST use the same line_buffer_idx value. The toggle happens ONCE after WeightLoader — NEVER between DataLoader and WeightLoader.
- Origin: sd_codegen uses separate DataLoaderManager and WeightLoaderManager, each starting at 0 and toggling independently — they always stay in sync. Our implementation replicates this with a single shared counter.
- P0 bug: An earlier implementation toggled between DL and WL, causing wrong line_buffer_idx in WL → mismatch with golden.
- Code: emitter.py, line 141 (toggle after both DL and WL)

**3. QuantLoader 1-based layer_idx (conv_layer_counter)**
- Rule: QuantLoader uses 1-based layer_idx that only increments for conv2d, deformable_conv2d, and offset_gen layers. prelu and pool2d layers do NOT increment the counter.
- EmitterState.conv_layer_counter starts at 0, incremented before each conv-class layer emission.
- Fix: Before fix, non-conv layers were incorrectly counted, producing non-sequential QuantLoader layer_idx values.

**4. src4 quirk in post_pass.py**
- Line 262: `code_dict["src4"] = src_code[2] if len(src_code) > 3 else 0`
- This assigns src_code[2] (not src_code[3]) to src4. This matches the golden sd_sr_codegen.py verbatim and must NOT be "fixed".

**5. torchvision.ops.deform_conv2d auto-conversion**
- torchvision.ops.deform_conv2d is automatically converted to nn.deformable_conv2d by relay.frontend.from_pytorch. No custom op registration needed.
- This was a key discovery: standard TVM from_pytorch handles this transparently.

**6. OffsetGenerator subgraph recognition**
- Pattern: layers[i].op=='pool2d' AND layers[i+1].op=='conv2d' AND layers[i+1].cout==18 AND layers[i+2].op=='deformable_conv2d'
- _OFFSET_GEN_COUT = 18 is the discriminating constant (2 × 9 offsets for 3×3 kernel)

**How to apply**: These findings are the core technical contributions of the paper. Each represents a non-obvious challenge that required careful analysis of TVM internals or hardware behavior.
