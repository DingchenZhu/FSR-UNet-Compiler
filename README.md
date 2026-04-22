# TVM Compiler Frontend for CNN Hardware Accelerator

A TVM-based compiler frontend that automatically compiles ONNX and PyTorch models into pseudo-instruction sequences for a custom CNN hardware accelerator.

> Chinese documentation: [README_CN.md](README_CN.md)

## Overview

The target accelerator features a MAC array, dual-path line buffers, and dedicated deformable convolution hardware (OffsetLoader + bilinear WeightLoader). Previously, code generation was done via hand-written Python scripts tightly coupled to individual models. This project replaces that with a programmable compiler frontend built on TVM Relay IR.

**Target models:**
- **USR-Net / UNet** — ONNX, 256×256, 50 layers, ~5000 instructions
- **FSRCNN** — PyTorch, includes OffsetGenerator + DeformableConv2d, 19 layers, 840 instructions

## Architecture

```
Model file (.onnx / .py)
        │
        ▼
┌───────────────┐
│   Frontend    │  relay.frontend.from_onnx / from_pytorch
│  frontend/    │  torchvision::deform_conv2d → nn.deformable_conv2d (automatic)
└───────┬───────┘
        │ Relay IRModule
        ▼
┌───────────────┐
│   Layer IR    │  Relay Call graph → LayerDesc list
│   ir/         │  + OffsetGenerator subgraph fusion (pool2d+conv2d → offset_gen)
└───────┬───────┘
        │ List[LayerDesc]
        ▼
┌───────────────┐
│    Tiling     │  LayerDesc → TilingPlan (tile sizes, addresses, ISA template params)
│   tiling/     │
└───────┬───────┘
        │ List[TilingPlan]
        ▼
┌───────────────┐
│    Backend    │  LayerDesc + TilingPlan → ISA instruction dicts
│   backend/    │  emitter.py (3 templates) + post_pass.py (deps + virtual regs)
└───────┬───────┘
        │ List[Dict]
        ▼
  Pseudo-instruction stream (pseudo_instructions.txt)
```

## Repository Layout

```
tvm-design/
├── pipeline.py              # End-to-end orchestration entry point
├── frontend/
│   ├── frontend.py          # ONNX / PyTorch model import
│   └── fsrcnn_loader.py     # FSRCNN get_model() wrapper
├── ir/
│   ├── layer_desc.py        # Relay IR → LayerDesc extraction
│   └── fusion_pass.py       # OffsetGenerator subgraph fusion pass
├── tiling/
│   └── tiling.py            # LayerDesc → TilingPlan
├── backend/
│   ├── isa.py               # 7 ISA instruction wrappers (golden-format compatible)
│   ├── emitter.py           # 3 emission templates (conv / deformable / offset_gen)
│   └── post_pass.py         # Dependency analysis + virtual register allocation
├── output/                  # Compiler outputs (relay_ir.txt, layer_descs.json, …)
└── docs/
    ├── record.md            # Development log
    └── compiler_roadmap.md  # Architecture roadmap
```

## Getting Started

### Requirements

- Python 3.9+
- TVM (with `relay.frontend`)
- PyTorch + torchvision (for PyTorch model path)
- ONNX (for ONNX model path)

### Compile an ONNX model

```bash
python3 pipeline.py \
    --model path/to/USR_Net.onnx \
    --type onnx \
    --input-shape 1 1 256 256 \
    --input-name data \
    --output-dir output/usr_net/ \
    --verbose
```

### Compile a PyTorch model (FSRCNN)

```bash
python3 pipeline.py \
    --model frontend/fsrcnn_loader.py \
    --type pytorch \
    --input-shape 1 1 36 64 \
    --output-dir output/fsrcnn/ \
    --verbose
```

### Multi-frame pipeline scheduling (load_next)

```python
from pipeline import run_pipeline, PipelineConfig

result = run_pipeline(
    model_path="path/to/model.onnx",
    model_type="onnx",
    input_shapes={"data": (1, 1, 256, 256)},
    config=PipelineConfig(
        output_dir="output/",
        is_first=True,           # emit 5-instruction DDR preload preamble
        load_next=True,          # prefetch next frame after layer-0 tiles complete
        image_transnum=576,      # 144×4 pixels per frame tile
        inter_layer_transnum=64, # offchip transfer at UNet→FSRCNN boundary
    ),
)
```

### Diff against a golden file

```bash
python3 pipeline.py --model ... --golden path/to/golden.txt --output-dir output/
```

## Key Design Decisions

### `ir/layer_desc.py` — Relay-to-LayerDesc extraction

Walks the Relay Call DAG and maps supported ops to `LayerDesc`:

| Relay op | `LayerDesc.op` |
|----------|----------------|
| `nn.conv2d` | `conv2d` |
| `nn.deformable_conv2d` | `deformable_conv2d` |
| `nn.max_pool2d` / `nn.avg_pool2d` | `pool2d` |
| `nn.relu` / `nn.prelu` | `relu` / `prelu` |

> **TVM ObjectRef identity**: DAG deduplication uses `expr in visited` (TVM's stable `__hash__` based on the C++ object pointer). Using `id(expr)` is incorrect — TVM creates a new Python wrapper on each access to the same underlying node, making `id()` unstable across accesses.

### `ir/fusion_pass.py` — OffsetGenerator fusion

Recognizes and fuses the `OffsetGenerator` subgraph (`AvgPool2d + Conv2d(cout=18)`) that precedes each deformable convolution:

```
pool2d → conv2d(cout=18) → deformable_conv2d
              ↓ fuse
         offset_gen      → deformable_conv2d
```

The fused `offset_gen` op uses a dedicated emission template that writes results to `dest_buffer_idx='offset_reg'`, which is consumed directly by the subsequent `OffsetLoader`. Without this pass, the offset conv would write to buffer `a` (wrong destination), producing incorrect deformable convolution results.

### `backend/emitter.py` — Three emission templates

| Template | Trigger | Instruction pattern |
|----------|---------|---------------------|
| Standard conv (Template A) | `op='conv2d'` | QuantLoader → per-W-tile × (DataLoader + WeightLoader + DataStorer) |
| Offset generator | `op='offset_gen'` | QuantLoader → 3×(DataLoader + WeightLoader) → DataStorer(dest=`offset_reg`) |
| Deformable conv | `op='deformable_conv2d'` | QuantLoader → H-steps × (ky × (OffsetLoader + ic×(DataLoader+WeightLoader)) + DataStorer) |

**`line_buffer_idx` invariant**: DataLoader and WeightLoader always receive the *same* `line_buffer_idx` value. A single toggle fires after WeightLoader, never between DataLoader and WeightLoader. This replicates the behavior of two independent managers both starting at 0 and toggling in lockstep.

**`QuantLoader` layer index**: 1-based, counting only `conv2d` / `deformable_conv2d` / `offset_gen` layers via a dedicated `conv_layer_counter` in `EmitterState`. Activation and pooling layers do not increment the counter.

### `backend/post_pass.py` — Post-processing

Seven dependency rules (ported verbatim from the golden `sd_sr_codegen.py`) followed by virtual register allocation (0/1 toggle for `line_buffer_idx`, `acc_reg_idx`, `quant_config_idx`). The `src4` quirk (assigns `src_code[2]` instead of `src_code[3]`) is preserved for golden parity.

## Verified Results

| Model | Layers | Instructions | Notes |
|-------|--------|-------------|-------|
| USR-Net (ONNX, 256×256) | 50 | 4995 | All conv2d; QuantLoader indices 1–28 |
| FSRCNN (PyTorch, 36×64) | 19 | 840 | 4× offset_gen + 4× deformable_conv2d |

FSRCNN instruction breakdown:

```
OffchipDataLoader:  1     QuantLoader:  12    DataLoader:  300
WeightLoader:     300     DataStorer:  112    OffsetLoader: 108
PseudoOp:           7     DataStorer(dest=offset_reg): 4  ✓
```

## Output Files

Each compilation writes to `--output-dir`:

| File | Contents |
|------|----------|
| `relay_ir.txt` | TVM Relay IR text dump |
| `layer_descs.json` | LayerDesc + TilingPlan per layer |
| `tiling_plan.json` | TilingPlan list |
| `pseudo_instructions.txt` | Instruction stream, one Python dict per line |

Example instruction format:

```python
{'code_num': [6], 'op_code': 'QuantLoader', 'quant_reg_load_idx': 0, 'quant_mode': 0, 'layer_idx': 1, 'transnum': 4, 'bas_addr': 0, ...}
{'code_num': [7], 'op_code': 'DataLoader', 'layer_idx': 0, 'line_buffer_reshape': 1, 'is_padding_row': 1, ...}
```
