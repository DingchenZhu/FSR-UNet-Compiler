# TVM Compiler Frontend for CNN Hardware Accelerator

面向自研 CNN 硬件加速器的 TVM 编译器前端，将 ONNX / PyTorch 模型自动编译为加速器伪指令序列。

## 项目背景

自研加速器含 MAC 阵列、双路 line buffer、可变形卷积硬件（OffsetLoader + 双线性 WeightLoader），原有 codegen 为手工编写 Python 脚本，与模型强绑定、无法复用。本项目基于 TVM Relay IR 构建可编程编译器前端，实现从模型到微指令的自动化映射。

目标模型：
- **USR-Net / UNet**（ONNX，256×256，50层，约5000条指令）
- **FSRCNN**（PyTorch，含 OffsetGenerator + DeformableConv2d，19层，840条指令）

## 架构概览

```
模型文件 (.onnx / .py)
      │
      ▼
┌─────────────┐
│   Frontend  │  relay.frontend.from_onnx / from_pytorch
│ frontend/   │  torchvision::deform_conv2d → nn.deformable_conv2d（自动）
└──────┬──────┘
       │ Relay IRModule
       ▼
┌─────────────┐
│  Layer IR   │  Relay Call 图 → LayerDesc 列表
│  ir/        │  + OffsetGenerator 子图融合（pool2d+conv2d → offset_gen）
└──────┬──────┘
       │ List[LayerDesc]
       ▼
┌─────────────┐
│   Tiling    │  每层 → TilingPlan（分块、地址、ISA 模板参数）
│  tiling/    │
└──────┬──────┘
       │ List[TilingPlan]
       ▼
┌─────────────┐
│   Backend   │  LayerDesc + TilingPlan → ISA 指令字典
│  backend/   │  emitter.py（3种模板）+ post_pass.py（依赖 + 虚拟寄存器）
└──────┬──────┘
       │ List[Dict]
       ▼
  伪指令序列（pseudo_instructions.txt）
```

## 目录结构

```
tvm-design/
├── pipeline.py              # 端到端编排入口
├── frontend/
│   ├── frontend.py          # ONNX / PyTorch 模型导入
│   └── fsrcnn_loader.py     # FSRCNN get_model() 封装
├── ir/
│   ├── layer_desc.py        # Relay IR → LayerDesc 提取
│   └── fusion_pass.py       # OffsetGenerator 子图融合 Pass
├── tiling/
│   └── tiling.py            # LayerDesc → TilingPlan
├── backend/
│   ├── isa.py               # 7 类 ISA 指令包装（golden 格式兼容）
│   ├── emitter.py           # 3 种指令模板（标准conv / deformable / offset_gen）
│   └── post_pass.py         # 依赖分析 + 虚拟寄存器分配
├── output/                  # 编译输出（relay_ir.txt / layer_descs.json 等）
└── docs/
    ├── record.md            # 开发日志
    └── compiler_roadmap.md  # 架构路线图
```

## 快速开始

### 依赖

- Python 3.9+
- TVM（含 `relay.frontend`）
- PyTorch + torchvision（PyTorch 模型路径）
- ONNX（ONNX 模型路径）

### 编译 ONNX 模型

```bash
python3 pipeline.py \
    --model path/to/USR_Net.onnx \
    --type onnx \
    --input-shape 1 1 256 256 \
    --input-name data \
    --output-dir output/usr_net/ \
    --verbose
```

### 编译 PyTorch 模型（FSRCNN）

```bash
python3 pipeline.py \
    --model frontend/fsrcnn_loader.py \
    --type pytorch \
    --input-shape 1 1 36 64 \
    --output-dir output/fsrcnn/ \
    --verbose
```

### 多帧流水调度（load_next）

```python
from pipeline import run_pipeline, PipelineConfig

result = run_pipeline(
    model_path="path/to/model.onnx",
    model_type="onnx",
    input_shapes={"data": (1, 1, 256, 256)},
    config=PipelineConfig(
        output_dir="output/",
        is_first=True,           # 发射 5 条 DDR 预加载 preamble
        load_next=True,          # 在 layer-0 结束后预取下一帧
        image_transnum=576,      # 144×4 像素/帧
        inter_layer_transnum=64, # UNet→FSRCNN 边界 offchip 传输
    ),
)
```

### 对比 Golden 文件

```bash
python3 pipeline.py --model ... --golden path/to/golden.txt --output-dir output/
```

## 核心模块说明

### `ir/layer_desc.py` — LayerDesc 提取

遍历 Relay Call 图，将支持的算子映射为 `LayerDesc`：

| Relay 算子 | LayerDesc.op |
|-----------|-------------|
| `nn.conv2d` | `conv2d` |
| `nn.deformable_conv2d` | `deformable_conv2d` |
| `nn.max_pool2d` / `nn.avg_pool2d` | `pool2d` |
| `nn.relu` / `nn.prelu` | `relu` / `prelu` |

> **重要**：DAG 去重使用 TVM 稳定哈希（`expr in visited`），不可用 `id(expr)`——TVM ObjectRef 每次访问同一 C++ 节点会产生新 Python 包装对象，`id()` 不稳定。

### `ir/fusion_pass.py` — OffsetGenerator 融合

识别并融合 `OffsetGenerator` 子图（`AvgPool2d + Conv2d(cout=18)`）：

```
pool2d → conv2d(cout=18) → deformable_conv2d
           ↓ 融合
offset_gen              → deformable_conv2d
```

融合后 `offset_gen` 走专用模板，DataStorer 写入 `dest_buffer_idx='offset_reg'`，供后续 OffsetLoader 消费。

### `backend/emitter.py` — 3 种指令模板

| 模板 | 触发条件 | 特征 |
|------|---------|------|
| 标准 conv（Template A） | `op='conv2d'` | QuantLoader → 每W宏块×(DL+WL+DS) |
| offset_gen | `op='offset_gen'` | QuantLoader → 3×(DL+WL) → DS(dest=offset_reg) |
| deformable conv | `op='deformable_conv2d'` | QuantLoader → H步×(ky×(OffsetLoader + ic×(DL+WL)) + DS) |

**line_buffer_idx 不变式**：DataLoader 和 WeightLoader 使用**同一**值，单次 toggle 在 WeightLoader 之后。两个独立 manager 从0开始并各自 toggle 与单共享计数器等价。

**QuantLoader layer_idx**：仅对 `conv2d` / `deformable_conv2d` / `offset_gen` 层计数，1-based 连续编号，prelu / pool 层不参与计数。

### `backend/post_pass.py` — 依赖分析与虚拟寄存器分配

7 条依赖规则（移植自 `sd_sr_codegen.py`）+ 虚拟 0/1 寄存器分配（`line_buffer_idx`, `acc_reg_idx`, `quant_config_idx`）。`src4` quirk 已保留（`src_code[2]` 而非 `src_code[3]`）。

## 验证结果

| 模型 | 层数 | 指令数 | 说明 |
|------|------|--------|------|
| USR-Net（ONNX, 256×256） | 50 | 4995 | 全 conv2d，QuantLoader 1–28 连续 |
| FSRCNN（PyTorch, 36×64） | 19 | 840 | 含4个 offset_gen + 4个 deformable_conv2d |

FSRCNN 指令分布：

```
OffchipDataLoader: 1   QuantLoader: 12   DataLoader: 300
WeightLoader: 300      DataStorer: 112   OffsetLoader: 108   PseudoOp: 7
DataStorer(dest=offset_reg): 4  ✓
```

## 输出文件格式

每次编译在 `--output-dir` 下生成：

| 文件 | 内容 |
|------|------|
| `relay_ir.txt` | TVM Relay IR 文本 |
| `layer_descs.json` | LayerDesc + TilingPlan 列表 |
| `tiling_plan.json` | TilingPlan 列表 |
| `pseudo_instructions.txt` | 伪指令序列，每行一个 Python dict |

伪指令格式示例：

```python
{'code_num': [6], 'op_code': 'QuantLoader', 'quant_reg_load_idx': 0, 'quant_mode': 0, 'layer_idx': 1, 'transnum': 4, 'bas_addr': 0, ...}
{'code_num': [7], 'op_code': 'DataLoader', 'layer_idx': 0, 'line_buffer_reshape': 1, 'is_padding_row': 1, ...}
```
