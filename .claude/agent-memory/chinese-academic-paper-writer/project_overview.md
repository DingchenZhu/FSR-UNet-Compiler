---
name: Project Overview
description: TVM compiler frontend for CNN hardware accelerator targeting FSRCNN and UNet models
type: project
---

**Fact**: This project implements a custom TVM-based compiler frontend for a proprietary CNN hardware accelerator. The golden reference is sd_sr_codegen.py.

**Why**: Standard TVM lowering cannot generate correct ISA instructions for the custom 7-instruction hardware (OffchipDataLoader, DataLoader, WeightLoader, OffsetLoader, QuantLoader, DataStorer, OffchipDataStorer). The project bridges TVM's high-level IR with hardware-specific instruction emission.

**Target models**:
- USR-Net / UNet: ONNX format, 256×256 input, ~50 layers, ~5000 instructions
- FSRCNN: PyTorch format, contains OffsetGenerator + DeformableConv2d, 19 layers (after fusion), 840 instructions

**Pipeline stages** (pipeline.py):
1. Frontend: relay.frontend.from_onnx / from_pytorch → Relay IRModule
2. Layer IR extraction (ir/layer_desc.py): Relay Call graph → LayerDesc list
3. Subgraph fusion pass (ir/fusion_pass.py): OffsetGenerator fusion
4. Tiling (tiling/tiling.py): LayerDesc → TilingPlan
5. Backend emission (backend/emitter.py): 3 instruction templates
6. Post-pass (backend/post_pass.py): dependency analysis + virtual register allocation

**How to apply**: When writing about this project, emphasize that the frontend is a purpose-built compilation stack, not a generic TVM deployment. Each stage is designed around hardware constraints.
