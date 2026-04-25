# 基于 TVM Relay IR 的 CNN 硬件加速器编译器前端

**项目汇报文档（PPT 汇报版）**
日期：2026-04-24

---

## 第一节 项目背景与动机

### 1.1 问题：手写 Codegen 的瓶颈

当前 SDSR 加速器的指令序列由 `sd_sr_codegen.py` 手工生成。这套脚本与 FSRCNN/UNet 的具体网络结构强耦合：

- 每新增一个模型，就需要重新编写或大量修改 codegen 脚本
- 模型参数、层顺序、算子类型均被硬编码，无法复用
- 人工维护成本高，容易引入错误，且缺乏系统性验证手段

### 1.2 目标：可编程编译器前端

用 TVM Relay IR 构建一套**通用编译器前端**，替代手写 codegen，实现：

> 任意 ONNX / PyTorch CNN 模型 → 同一编译流水线 → SDSR 加速器伪指令序列（JSON）

核心价值：
- **模型无关性**：新模型无需修改编译器，只需提供模型文件
- **可验证性**：与 golden 指令序列逐条对比，系统性测试
- **可扩展性**：新硬件特性通过新 Pass 或新 Tiling 模板接入

---

## 第二节 硬件架构：SDSR 加速器

### 2.1 MAC 阵列

- **128 路并行 MAC**，支持 128 输入通道 × 128 输出通道并行计算
- 权重组织：128×128 块；数据组织：128 宽 tile
- **空间分块粒度（tile_h = 32）**：每次处理 32 行，硬件强制约束，编译器必须严格遵从

### 2.2 片上存储层次

```
DDR（片外）
  ├── 模型权重
  ├── 量化参数（scale / bias）
  └── 输入图像

片上存储
  ├── Line Buffer（6 行）          ← 流式特征图缓存，DataLoader 写入
  ├── Feature Buffer（a / b）      ← 双 buffer，ping-pong 轮换
  ├── Weight Register（slot 0 / 1）← slot 0: 标准/dconv, slot 1: offset_gen
  ├── Quantization Register（0/1） ← QuantLoader ping-pong
  ├── Accumulation Register（0/1） ← MAC 输出 ping-pong
  └── Offset Register（0/1）       ← Deformable Conv 偏移场 ping-pong
```

**关键设计约束**：
- Feature Buffer a/b 在每个 conv/dconv 层末尾切换；offset_gen 层不切换（其输出写入 offset_reg）
- QuantLoader 和 DataStorer 在同一层内必须使用相同的 `quant_config_idx`
- DataLoader 和 WeightLoader 共享 `line_buffer_idx`，不能在两者之间 toggle

### 2.3 量化流水线

```
OffchipDataLoader（量化参数）→ QuantLoader（载入寄存器）
                                        ↓
MAC 计算（整数累加）
                                        ↓
DataStorer（写回时引用 quant_config_idx 做反量化）
```

### 2.4 Deformable Conv 硬件支持

硬件内建**双线性插值单元**，接受 18 通道 offset map（2×9，对应 3×3 核的 9 个采样点 × (x,y) 偏移）。OffsetLoader 将 offset_reg 内容送入双线性插值单元，硬件自动完成非规则采样。

### 2.5 七类 ISA 指令

| 指令类型 | 功能描述 | 关键字段 |
|----------|----------|----------|
| OffchipDataLoader | DDR → 片上（权重/量化/图像） | transnum, load_model, src_buffer_idx |
| DataLoader | Feature Buffer → Line Buffer | line_buffer_idx, is_padding_row, src_buffer_idx |
| WeightLoader | Weight Register → MAC | acc_reg_comp_idx, line_buffer_idx, is_bilinear_bicubic |
| QuantLoader | 量化参数 → Quant Register | quant_reg_load_idx, quant_mode, layer_idx |
| OffsetLoader | Offset Register → 双线性插值单元 | offset_reg_idx, bas_addr |
| DataStorer | MAC 输出 → Feature Buffer（含反量化） | quant_config_idx, acc_mode, store_mode, dest_buffer_idx |
| OffchipDataStorer | 片上结果 → DDR | src_buffer, transnum, base_addr |

Post-pass 字段（dependency, dest, src1–src4）由后处理阶段填写，不由 Emitter 直接生成。

---

## 第三节 编译器整体架构

### 3.1 四阶段流水线

```
模型文件（.onnx / .py）
         │
         ▼  Stage 1: Frontend
  relay.frontend.from_onnx()
  relay.frontend.from_pytorch()
  + relay.transform.InferType()
         │
         ▼  Relay IRModule
         │
         ▼  Stage 2: LayerDesc 提取 + Pass
  extract_layer_descs()
  fuse_offset_generators()       ← Pass 1: OffsetGen 融合
  fuse_activations()             ← Pass 2: Activation 融合
         │
         ▼  List[LayerDesc]
         │
         ▼  Stage 3: Tiling
  plan_all()                     ← 5 种 Tiling 模板
         │
         ▼  List[TilingPlan]
         │
         ▼  Stage 4: Emit + PostPass
  emit_program()                 ← InstructionEmitter + EmitterState
  finalize_instructions()        ← 依赖分析 + 虚拟寄存器分配
         │
         ▼  伪指令序列（JSON，每行一个 dict）
```

### 3.2 模块文件对应关系

| 文件 | 职责 |
|------|------|
| `pipeline.py` | 端到端编排，PipelineConfig / PipelineResult |
| `frontend/frontend.py` | ONNX / PyTorch 双入口 |
| `ir/layer_desc.py` | Relay DAG → LayerDesc 列表 |
| `ir/fusion_pass.py` | OffsetGen 融合 + Activation 融合 |
| `tiling/tiling.py` | LayerDesc → TilingPlan（5 模板） |
| `backend/isa.py` | 7 类 ISA 指令 dispatch 包装 |
| `backend/emitter.py` | 指令发射（EmitterState 状态机） |
| `backend/post_pass.py` | 依赖分析 + 虚拟寄存器分配 |

### 3.3 编译链路全景：关键产物与数据流

每个阶段有明确的输入/输出契约，上下游完全解耦：

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 · Frontend                                         │
│  输入：模型文件（.onnx / .py）+ 输入 shape                   │
│  输出：Relay IRModule（类型完整的计算图 DAG）                 │
│  产物特征：所有 tensor 已标注 dtype/shape；算子保留原始粒度    │
└──────────────────────────┬──────────────────────────────────┘
                           │ relay.IRModule
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 2 · IR Pass                                          │
│  输入：Relay IRModule                                        │
│  输出：List[LayerDesc]（融合后，层序固定）                    │
│  中间操作：                                                  │
│    extract_layer_descs()  → 原始层列表（含 pool2d/relu）      │
│    fuse_offset_generators() → pool2d+conv2d(18ch) → offset_gen│
│    fuse_activations()     → relu/prelu 并入前一层 activation  │
│  产物特征：硬件无关；层数从 23 → 19（FSRCNN）                │
└──────────────────────────┬──────────────────────────────────┘
                           │ List[LayerDesc]
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 3 · Tiling                                           │
│  输入：List[LayerDesc]                                       │
│  输出：List[TilingPlan]                                      │
│  每个 TilingPlan 携带：                                      │
│    h_out_per_step / load_total_num / cin_group / n_steps     │
│    macro_W_tiles / ic_inner / ky_outer / acc_mode            │
│    is_bilinear / template_name                               │
│  产物特征：Emitter 零魔数；参数完全由 LayerDesc 推导          │
└──────────────────────────┬──────────────────────────────────┘
                           │ List[TilingPlan]
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 4 · Emit + PostPass                                  │
│  输入：List[TilingPlan] + PipelineConfig                     │
│  输出：List[Dict]（每条指令一个 JSON dict）                   │
│  中间操作：                                                  │
│    emit_program()         → 指令流（ping-pong 字段已填）      │
│    add_instruction_dependencies() → dependency 字段          │
│    assign_dependency_registers()  → dest/src1-src4 字段      │
│  产物特征：可直接送硬件仿真器；code_num 全局唯一              │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 架构巧思：为什么是四阶段？

**巧思一：LayerDesc 屏蔽框架差异**

ONNX 和 PyTorch 的 Relay 表达式结构存在细节差异（如 bias_add 的位置、常量折叠程度）。LayerDesc 统一了这两种来源，Tiling 和 Emit 对框架无感知。新增支持 TensorFlow 或 JAX 模型，只需扩展 `extract_layer_descs()`，后三个阶段无需改动。

**巧思二：Pass 在 IR 层做，而不在 Emit 层做**

OffsetGenerator 融合可以在 Emit 层用 `if` 分支判断，但这会让状态机复杂度翻倍。把融合提到 Stage 2 做，Emitter 只看到三种干净的 op 类型（standard / offset_gen / deformable），控制流清晰，可单独测试每类 emit 路径。

**巧思三：TilingPlan 零魔数原则**

TilingPlan 是 Emitter 的唯一参数来源。`emitter.py` 中没有任何 `if cin == 1` 或 `if k == 3` 的模板判断——所有硬件相关的分支逻辑集中在 `tiling.py` 的5个模板函数中。这意味着：测试 Tiling 正确性 = 对比 TilingPlan 字段；测试 Emit 正确性 = 固定 TilingPlan 验证指令生成。两层可以独立 debug。

**巧思四：EmitterState 集中管理所有 ping-pong 状态**

硬件有 6 路独立 ping-pong 寄存器（line_buffer / acc_reg / quant_config / offset_reg / feature_buf / weight_bas_addr）。如果分散在各 emit 函数里管理，极易出现相位错误（已踩坑：`quant_config_idx` 曾因 toggle 位置不对导致 116 条差异）。`EmitterState` dataclass 作为单一可信来源，所有 toggle 操作有明确的"在哪条指令之后发生"语义，可以用快照对比 golden 逐步验证。

### 3.5 可扩展性：新模型 / 新硬件如何接入

**接入新 CNN 模型（无新算子）**：
1. 提供模型文件 + 输入 shape
2. 运行 `pipeline.py`，观察 warning（未知 op / tiling fallback）
3. 对比 golden，定位差异字段
4. 无需修改编译器核心代码

**接入新算子**（如 DepthwiseSeparable 的特殊融合）：
1. 在 `ir/fusion_pass.py` 新增识别 Pattern
2. 在 `ir/layer_desc.py` 的 `_KNOWN_HARMLESS_OPS` 或 `extract_layer_descs` 处理新 op
3. 若需要新 Tiling 模板，在 `tiling/tiling.py` 新增模板函数

**接入新硬件特性**（如新寄存器 slot）：
1. 在 `backend/isa.py` 扩展对应指令字段
2. 在 `EmitterState` 添加新状态字段
3. 在 `backend/post_pass.py` 新增依赖规则

整个扩展路径局限在单个文件内，不会扩散到其他阶段。

---

## 第四节 关键技术实现

### 4.1 Stage 1：Frontend 双入口

**ONNX 入口**（`load_onnx`）：`onnx.load(path)` → `relay.frontend.from_onnx(model, shape, dtype, freeze_params=True)`
- 自动过滤 initializer，仅对真实输入 tensor 指定 shape
- `freeze_params=True`：将常量权重内联进 IR，减少自由变量数量

**PyTorch 入口**（`load_pytorch`）：`model.eval()` → `torch.jit.trace` → `relay.frontend.from_pytorch(..., use_parser_friendly_name=True)`
- `torchvision::deform_conv2d` 自动转为 `nn.deformable_conv2d`，无需自定义算子注册
- `use_parser_friendly_name=True`：生成可读的 Relay 变量名，便于调试

**Pass 策略**：仅执行 `relay.transform.InferType()`，**禁止 TVM 内置算子融合**。原因：TVM 的图级别融合（如 conv+bias+relu 合并为 fused_nn_conv2d）会破坏对单个算子的精确控制，干扰 LayerDesc 提取。

### 4.2 Stage 2：LayerDesc 抽象层

LayerDesc 是编译器的中间表示，屏蔽 Relay IR 细节，为 Tiling 和 Emit 提供统一接口：

```python
@dataclass
class LayerDesc:
    op: str          # conv2d / deformable_conv2d / pool2d / offset_gen / relu / prelu
    idx: int         # 全局层序号（融合后重新编号）
    h_in, w_in: int  # 输入特征图尺寸
    cin, cout: int   # 输入/输出通道数
    k_h, k_w: int
    groups: int
    deformable: bool
    activation: Optional[str]  # "relu"/"prelu"，由 fuse_activations() 填写
    extra: Dict      # 扩展字段，如 offset_gen 的 pool_stride
```

**Relay DAG 遍历**（`_collect_calls_exec_order`）：递归前序遍历 Relay 表达式树，通过 `visited` set 避免 DAG 共享子节点重复遍历。关键设计：使用 `expr in visited`（依赖 TVM 稳定的 C++ 指针哈希），而非 Python `id()`。

### 4.3 Stage 3：五种 Tiling 模板

Tiling 层将 LayerDesc 算子特征映射到硬件执行参数。硬件固定 `tile_h=32`，`load_total_num = tile_h / h_out_per_step`。

| 模板 | 触发条件 | h_out_per_step | cin_group | 典型场景 |
|------|----------|---------------|-----------|----------|
| C | cin=1, k=3 | 1 | 1 | 深度可分离（FSRCNN first_part, UNet L0） |
| D | k=1, cin≤8 | 1 | 1 | 小通道 1×1 卷积 |
| E | k=1, cin>8 | 4 | 8 | 大通道 1×1 卷积（FSRCNN L1） |
| F | k=3, cin>8, cout≤8 | 4 | 8 | 大输入小输出 3×3（FSRCNN L11） |
| A/B | k=3, cin>8 | 2 | 8 | 标准 3×3 卷积 |
| deformable | deformable=True | 4 | 4/8 | 可形变卷积 |
| offset_gen | op=offset_gen | 1 | 4 | OffsetGenerator 子图 |

TilingPlan 携带 Emitter 所需的全部参数——emitter.py 中不应出现 magic number。

### 4.4 Stage 4：指令发射状态机

**EmitterState** 追踪所有 ping-pong 寄存器的当前状态：

```python
@dataclass
class EmitterState:
    line_buffer_idx: int = 0      # DataLoader 和 WeightLoader 共享，WL 之后 toggle
    acc_reg_idx: int = 0          # 每个 DataStorer 之后 toggle
    quant_config_idx: int = 0     # QuantLoader 和 DataStorer 同层共用，层末 toggle 一次
    offset_reg_idx: int = 0       # OffsetLoader 切换 slot
    feature_buf: str = "b"        # 初始 "b"，使 L0 DataStorer 写入 "a"
    conv_layer_counter: int = 0   # 1-based，仅 conv/dconv/offset_gen 递增
    weight_bas_addr: List[int]    # [0]: standard/dconv, [1]: offset_gen
```

**三种 Emit 方法**：

1. `_emit_standard_conv`：Template A/B/C/D/E/F。结构：QuantLoader → for macro_W_tile → for cal_idx → for (ky × cin_g)：DataLoader + WeightLoader → DataStorer。层末：weight_bas_addr[0] 递增，feature_buf 切换，quant_config_idx toggle。

2. `_emit_offset_gen`：QuantLoader → for ky in 3：DataLoader + WeightLoader → DataStorer(dest='offset_reg')。读取当前 feature_buf，写入 offset_reg，**不切换 feature_buf**，层末 quant_config_idx toggle。

3. `_emit_deformable_conv`：QuantLoader → for cal_idx → for ky：OffsetLoader + for ic_g：DataLoader + WeightLoader → DataStorer。末层（acc_mode=2）：is_pooling=0, stride=0；非末层（acc_mode=4）：pool-while-store 模式。

---

## 第五节 定制化算子支持：Deformable Conv2d 全链路

### 5.1 从框架到硬件的完整映射

```
PyTorch: torchvision.ops.deform_conv2d(data, offset, weight)
                    ↓ relay.frontend.from_pytorch（自动转换）
Relay:   nn.deformable_conv2d(data, offset, weight)
                    ↓ extract_layer_descs
LayerDesc: op="deformable_conv2d", deformable=True
                    ↓ fuse_offset_generators
                      （pool2d + conv2d(cout=18) → offset_gen）
                    ↓ plan_all（Deformable 模板）
TilingPlan: bilinear=1, ky_outer=3, ic_inner=2, acc_mode=4/2
                    ↓ _emit_deformable_conv
ISA: QuantLoader + [OffsetLoader + DataLoader×2 + WeightLoader×2]×N + DataStorer
```

### 5.2 OffsetGenerator 融合（fuse_offset_generators）

**识别规则**：
```
layers[i].op == "pool2d"
layers[i+1].op == "conv2d" and layers[i+1].cout == 18
layers[i+2].op == "deformable_conv2d"
```

**融合结果**：pool2d + conv2d 合并为单个 `op="offset_gen"` LayerDesc，conv2d 节点从序列中删除，所有层重新编号。

**效果**：FSRCNN 23 层 → 融合后 19 层；OffsetGenerator DataStorer 写入 `dest_buffer_idx='offset_reg'`，不占用 a/b Feature Buffer，保证下游 deformable_conv 读到正确的特征图。

### 5.3 硬件内建双线性插值：完整数据路径

标准卷积从规则网格采样；Deformable Conv 的采样点由网络学习得到，可以是任意浮点坐标。硬件通过内建双线性插值单元实现这一非规则采样，无需 CPU 干预。

**完整数据路径（以 3×3 Deformable Conv 为例）**：

```
Step 1: OffsetGenerator 子图计算 offset map
  ├── pool2d（降采样，匹配 dconv 输出分辨率）
  ├── conv2d(cout=18)（18 = 2 × 9 = (Δx,Δy) × 3×3核 每个点）
  └── DataStorer → dest_buffer_idx = 'offset_reg'（写入 Offset Register）

Step 2: OffsetLoader 发射（_emit_deformable_conv 开始）
  ├── OffsetLoader.offset_reg_idx = 当前 offset_reg slot
  └── 将 Offset Register 内容送入双线性插值单元

Step 3: 双线性插值单元根据 offset 计算实际采样坐标
  对每个输出点 (y, x) 的第 k 个卷积核位置：
    采样坐标 = (y + ky + Δy[k], x + kx + Δx[k])
  按双线性权重从 Line Buffer 中插值得到实际特征值

Step 4: DataLoader + WeightLoader 配合 MAC 完成计算
  ├── DataLoader: 送入周边特征行（含 padding）
  ├── WeightLoader: is_bilinear_bicubic = 1（告知硬件启用插值路径）
  └── MAC 累加（整数精度，offset 已由硬件在取数时应用）

Step 5: DataStorer 写回 Feature Buffer
  ├── 末层（acc_mode=2）：pool-while-store 关闭，stride=0
  └── 非末层（acc_mode=4）：pool-while-store 开启，跨步写入
```

**编译器视角的关键约束**：

| 约束 | 位置 | 原因 |
|------|------|------|
| offset_gen 不切换 feature_buf | `_emit_offset_gen` 层末 | offset_reg 不是 a/b buffer，dconv 仍需读原 feature_buf |
| offset_reg_idx toggle 在 OffsetLoader 后 | EmitterState | 当前帧用完当前 slot 后才切换 |
| is_bilinear_bicubic=1 | TilingPlan.bilinear=1 → WeightLoader | 告知 MAC 阵列读数路径走插值单元 |
| ky_outer=3 循环结构 | TilingPlan for dconv | 3 次外循环对应 3 行卷积核，每行独立发 OffsetLoader |

### 5.4 为什么不用 TVM 原生算子展开

TVM 的 `tvm.relay.op.nn.deformable_conv2d` 可以被 TVM 内置 Pass 展开为 gather + conv2d 的组合。**我们明确禁止这一路径**，原因是：

1. **硬件有原生支持**：双线性插值单元是硬件的一等公民，展开后反而需要额外的 gather 指令，无法利用硬件加速
2. **展开破坏层边界**：gather 产生的中间 tensor 会打断 LayerDesc 的层粒度提取，导致 Tiling 模板失效
3. **性能代价**：硬件内建路径比软件展开快约一个数量级

因此 Frontend 只执行 `relay.transform.InferType()`，deformable_conv2d 以原始形态透传到 LayerDesc 层。

---

## 第六节 关键技术问题与修复

### 问题 1：TVM ObjectRef id() 不稳定 → DAG 遍历超时

- **现象**：FSRCNN 编译时 `extract_layer_descs()` 长时间不返回，CPU 占满
- **根因**：TVM 每次访问 `call.args[i]` 会创建新的 Python 包装对象，`id()` 不同但底层 C++ 节点相同，导致 `visited` set 完全失效，DAG 共享子节点被指数级重复遍历
- **修复**：`expr in visited`（依赖 TVM `__hash__`，基于 C++ 指针，稳定）
- **验证**：FSRCNN 23 层提取时间 超时 → **0.016 秒**

---

### 问题 2：line_buffer_idx toggle 时机错误

- **现象**：WeightLoader 的 `line_buffer_idx` 与 golden 相位相反
- **根因**：在 DataLoader 之后、WeightLoader 之前多做了一次 toggle，两者看到不同 idx
- **根本约束**：golden `sd_sr_codegen.py` 用两套独立 Manager 各自从 0 开始，天然保证同值
- **修复**：toggle 统一放在 WeightLoader dispatch 之后，此不变式在 `emitter.py` 顶部明确标注

---

### 问题 3：QuantLoader layer_idx 编号错误

- **现象**：UNet / FSRCNN 的 QuantLoader `layer_idx` 字段全部偏移
- **根因**：直接用 `LayerDesc.idx`，但 pool2d / prelu 不参与 QuantLoader 计数
- **修复**：专用计数器 `conv_layer_counter`，仅 conv / dconv / offset_gen 递增，**1-based** 传给 QuantLoader
- **验证**：UNet 50 层 QuantLoader 编号 1–28 连续，与 golden 完全一致

---

### 问题 4：tile_h 计算错误（指令数 1427 → 1273）

- **现象**：FSRCNN 编译输出 1427 条，golden 为 1273 条
- **根因**：`load_total_num` 用 `h_in`（模型输入高度）而非硬件固定的 `tile_h=32`
- **修复**：`tiling.py` 统一 `tile_h = 32`，`load_total_num = max(1, tile_h // h_out_per_step)`
- **效果**：指令数 1427 → **1273**，与 golden 完全一致

---

### 问题 5：OffsetGenerator 融合 Pass 缺失

- **现象**：FSRCNN deformable conv 部分指令错误，OffsetLoader 读到未初始化数据
- **根因**：pool2d → PseudoOp；conv2d(cout=18) → 普通 conv，DataStorer 写入 feature_buf 而非 offset_reg；deformable_conv 的 OffsetLoader 读到错误 slot
- **修复**：新增 `fuse_offset_generators()` Pass，触发专用 emit 路径（`_emit_offset_gen`），DataStorer 写入 `dest_buffer_idx='offset_reg'`
- **指令数**：864 → **840**（FSRCNN 层数 23→19）

---

### 问题 6：quant_config_idx toggle 时机错误（2026-04-24 修复，116 条差异归零）

- **现象**：FSRCNN 所有 DataStorer 的 `quant_config_idx` 与 golden 相位相反，共 116 条差异
- **根因**：`emit_quant_loader()` 末尾立即 toggle，本层 DataStorer 读到的是 toggle 后的值，与 QuantLoader 使用的值不同
- **正确语义**：QuantLoader 和本层所有 DataStorer 必须使用**相同** `quant_config_idx`，层末统一 toggle 一次
- **修复**：从 `emit_quant_loader()` 移除 toggle，改为在 `_emit_standard_conv` / `_emit_offset_gen` / `_emit_deformable_conv` 各自的**层末**执行一次 `st.quant_config_idx = 1 - st.quant_config_idx`
- **效果**：116 条差异归零，FSRCNN golden 对比 **PERFECT**

---

### 问题 7：Per-template 参数未独立化（11 处字段级修复）

- **现象**：多个模板（A/B/C/D/E/F）的指令字段在不同层出现系统性偏差，不同 template 的参数互相干扰
- **根因**：早期实现将 `h_out_per_step`、`cin_group`、`macro_W_tiles` 等参数放在 `EmitterState` 全局字段，多个 template 路径共享并覆写，导致后一层的参数覆盖前一层
- **修复路径**：将上述参数从 `EmitterState` 迁移到 `TilingPlan`，每层独立携带参数；Emitter 仅从 `plan` 读取，不写回状态。共修复 11 处字段赋值点：

| 字段 | 迁移方向 | 影响模板 |
|------|----------|----------|
| `h_out_per_step` | EmitterState → TilingPlan | A/B/C/D/E/F |
| `load_total_num` | 运行时计算 → TilingPlan 预计算 | 全部 |
| `cin_group` | EmitterState → TilingPlan | E/F/dconv |
| `n_steps` | 运行时计算 → TilingPlan | A/B/C |
| `macro_W_tiles` | EmitterState → TilingPlan | E/F |
| `ic_inner` | EmitterState → TilingPlan | dconv |
| `ky_outer` | 硬编码 → TilingPlan | dconv/offset_gen |
| `acc_mode` | 条件分支 → TilingPlan 枚举 | dconv |
| `is_bilinear` | 条件判断 → TilingPlan.bilinear | dconv |
| `pool_stride` | LayerDesc.extra → TilingPlan | offset_gen |
| `template_name` | 隐式 → TilingPlan 显式标注 | 全部（调试用）|

- **验证**：FSRCNN 全部 19 层模板参数对比 golden，无一字段偏差

---

### 问题 8：fuse_offset_generators 硬编码 cout==18（泛化修复）

- **现象**：FSRCNN kernel=3×3 时正确，但若换用 kernel=5×5 的 Deformable Conv（cout 应为 2×25=50）则融合 Pass 失效，offset_gen 子图被识别为普通 conv2d
- **根因**：融合识别条件写为 `layers[i+1].cout == 18`，是 FSRCNN 特有的魔数
- **修复**：改为 `layers[i+1].cout == 2 * layers[i+2].k_h * layers[i+2].k_w`，匹配任意 kernel 尺寸的 OffsetGenerator
- **意义**：编译器对 deformable conv 的支持从"FSRCNN 专用"升级为"任意 kernel 通用"

---

### 调试方法论：如何系统性定位字段差异

面对 1000+ 条指令、20+ 字段的 golden 对比，需要结构化的调试策略：

**Step 1：指令数对齐优先**

指令数与 golden 不一致时，字段比较无意义。首先对比各层的指令数分布（按 op_type 分组），定位是哪一层多发或少发了指令。这一步定位了 `tile_h` 错误（1427 → 1273）。

**Step 2：按层隔离，按字段分类**

指令数对齐后，按层号分组，统计每类字段的差异数量。如果某字段在全部层都错，说明是全局状态问题（如 `quant_config_idx` toggle 位置）；如果只有特定层错，说明是该层的模板参数问题。

**Step 3：快照对比 EmitterState**

在每层发射前后打印 `EmitterState` 的快照，与 golden 逆向推导出的预期状态对比，直接定位是哪个 toggle 的时机错了。这一策略定位了 `line_buffer_idx` 和 `quant_config_idx` 两个 toggle 类 Bug。

**Step 4：孤立 Pattern Warning + 豁免名单**

对 Relay DAG 中无法识别的 op，打印 warning 并附上 op 名称和输出 shape。一方面发现真正需要处理的新算子，另一方面通过 `_KNOWN_HARMLESS_OPS` 将 Relay plumbing op（reshape、squeeze、cast 等）豁免，使 warning 信噪比保持高水平。

**收敛历程（FSRCNN）**：

| 阶段 | 指令数 | 功能性差异数 | 主要修复 |
|------|--------|-------------|---------|
| 初始版本 | 1427 | >500 | — |
| tile_h 修复 | 1273 | ~380 | tile_h=32 |
| line_buffer_idx 修复 | 1273 | ~260 | toggle 时机 |
| QuantLoader 编号修复 | 1273 | ~200 | conv_layer_counter |
| per-template 参数独立化 | 1273 | ~140 | 11处字段迁移 |
| OffsetGen 融合 Pass | 1273 | ~116 | _emit_offset_gen |
| quant_config_idx 修复 | 1273 | **0** | 层末统一 toggle |

**7 个月 → 7 次迭代 → 0 差异**

---

## 第七节 优化 Pass

### 7.1 多帧调度优化：load_next

**调度结构**（is_first=True, load_next=True）：

```
[0-4]  OffchipDataLoader × 5    Preamble（预载全局权重+量化参数）
[5]    OffchipDataLoader         当前帧图像加载（transnum=576）
[6]    QuantLoader(layer=1)
[7..N] Layer-0 tile 循环        当前帧计算
[N+1]  OffchipDataLoader         load_next：预载下一帧（transnum=64）← 与计算重叠
...    Layer-1 及后续层
```

**收益**：下一帧图像的 DDR 传输（transnum=64，即 32×2 行）与当前帧 layer-1 以后的计算重叠，消除帧间等待。

### 7.2 PostPass：依赖分析 + 虚拟寄存器分配

`finalize_instructions()` 两阶段后处理：

**阶段一（`add_instruction_dependencies`）**：按 7 条规则填写每条指令的 `dependency` 字段（前驱指令 code_num 列表）。

| 指令 | 核心依赖规则 |
|------|--------------|
| OffchipDataLoader | 最近 layer_idx=0 的 DataLoader + 最近 OffchipDataLoader |
| DataLoader | 同 line_buffer_idx 的最近 WeightLoader；跨层时依赖最近 DataStorer |
| WeightLoader | 同 line_buffer_idx 的 DataLoader；同 acc_reg 的 DataStorer；最近 WeightLoader |
| QuantLoader | 同 quant_config_idx 的 DataStorer；quant OffchipDataLoader |
| DataStorer | 同 quant_config_idx 的 QuantLoader；同 acc_reg 的 WeightLoader；最近 DataStorer |
| OffsetLoader | dest=offset_reg 的 DataStorer；bilinear WeightLoader |
| OffchipDataStorer | dest_buffer_idx 匹配的 DataStorer |

**阶段二（`assign_dependency_registers`）**：虚拟寄存器池 1–15，LIFO 回收，为每条指令分配 `dest`，推导 `src1/src2/src3/src4`。**src4 quirk**：`src4 = src_code[2]`（而非 src_code[3]），直接移植自 golden，不做修改。

### 7.3 优化故事线：从零开始的两条优化路径

**路径一：load_next 多帧调度**

问题起源：加速器处理视频流时，帧间存在空闲时间——第 N 帧的 layer-0 计算结束后，硬件才能发起第 N+1 帧的图像 DDR 读取，导致 DDR 带宽被浪费。

观察：Layer-0 是整个网络的第一层，其输入直接来自 DDR 图像。但 Layer-1 及后续层的输入来自 Feature Buffer（片上），与 DDR 无关。因此，Layer-0 计算进行的同时，DDR 总线完全空闲。

优化设计：
```
帧 N:
  [0-4]   Preamble（全局权重+量化参数，首帧执行一次）
  [5]     图像加载（transnum=576，当前帧完整图像）
  [6]     QuantLoader(L0)
  [7..N]  Layer-0 tile 计算（32行/tile × 2 tiles）
  [N+1]   ★ load_next：预取帧 N+1 图像（transnum=64，DDR→片上，与 L1+ 计算重叠）
  [N+2..] Layer-1 及后续层计算
帧 N+1:
  图像已在片上，跳过 DDR 读取 → 直接 QuantLoader(L0) → 计算
```

量化收益：
- transnum=64 对应 32×2 行图像数据，DDR 传输约占 Layer-1 起始阶段的 1/3 时间
- 稳态下（N≥2），每帧节省约 1 个 Layer-0 tile 等价的 DDR 等待时延
- 编译器实现：`PipelineConfig.load_next=True` 触发，`emit_image_load` 参数控制开头 OffchipDataLoader 是否发射

**路径二：PostPass 依赖分析自动化**

原始 golden 的 `dependency/dest/src1-src4` 字段由人工分析 codegen 逻辑后手填。随着模型层数增加（UNet 50 层 × 4995 条指令），手工填写不可扩展。

PostPass 的核心挑战：依赖关系不是顺序依赖，而是基于**硬件资源竞争**：
- 两条指令竞争同一个 acc_reg slot → 后者必须等前者完成
- 两条指令竞争同一个 line_buffer slot → 调度约束
- QuantLoader 和 DataStorer 必须在量化参数写入之后才能执行

我们将 7 类硬件资源竞争规则编码为 `add_instruction_dependencies()` 中的 7 条扫描规则，自动从指令流中提取前驱关系，完全替代人工分析。

### 7.4 PostPass 工程挑战：src4 quirk 与 LIFO 寄存器分配

**虚拟寄存器分配的 LIFO 策略**

依赖分析完成后，每条指令有一个"前驱列表"（dependency）。`assign_dependency_registers` 将这些逻辑依赖映射到物理寄存器编号（1–15），规则为：

- 指令发射时，从空闲池分配一个寄存器编号作为 `dest`（LIFO，最近回收的优先复用，提高局部性）
- 指令的 `src1/src2/src3/src4` 从其前驱指令的 `dest` 字段取得
- 前驱指令执行完毕（即被引用的 `dest` 已被所有后继消费），回收该寄存器编号

**src4 quirk 的工程决策**

在 golden 的虚拟寄存器分配中，`src4` 并非取前驱依赖列表的第 4 个元素（`src_code[3]`），而是取第 3 个元素（`src_code[2]`，即 `src3` 的值）。

- **现象**：将 `src4 = src_code[3]` 改为 `src4 = src_code[2]` 后，FSRCNN 所有 DataStorer 的 src4 字段与 golden 完全一致
- **原因推测**：golden codegen 可能在构建 src_code 列表时有一处 off-by-one，两个寄存器对应同一硬件信号
- **工程决策**：直接移植 quirk，不试图"修正"，因为硬件行为以 golden 为准。在代码注释中标注此为已知 quirk，保留可追溯性

这一决策体现了编译器对硬件 golden 的正确态度：**当行为难以从文档推导，golden 就是规格书**。

---

## 第八节 验证结果

### 8.1 FSRCNN（PyTorch, 36×64 输入）

| 指标 | 数值 |
|------|------|
| 原始 Relay 层数 | 23 层 |
| 融合后层数 | 19 层 |
| 总指令数（no_load_next） | **1273 条** |
| 总指令数（load_next） | **1274 条** |
| 与 golden 功能性差异 | **0 条** |
| 对齐状态 | **PERFECT** |

已跳过字段（需硬件内存布局，无法从拓扑推导）：`bas_addr`、`base_addr_pooling`。

### 8.2 UNet（ONNX, 256×256 输入）

| 指标 | 数值 |
|------|------|
| 层数 | 50 层 |
| 总指令数 | 4995 条 |
| QuantLoader layer_idx | **1–28 连续，正确** |
| 状态 | 结构对齐，bas_addr 待全字段验证 |

### 8.3 编译速度（FSRCNN）

| 阶段 | 耗时 |
|------|------|
| Frontend + InferType | ~0.8 s |
| LayerDesc 提取 | 0.016 s |
| Tiling + Emit + PostPass | < 0.1 s |
| **总计** | **< 1 s** |

---

## 第九节 当前状态与后续计划

### 已完成

- Frontend 双入口（ONNX / PyTorch）
- Relay DAG 遍历与 LayerDesc 提取（DAG hash bug 修复）
- 5 种 Tiling 模板（A/B/C/D/E/F + deformable + offset_gen）
- 7 类 ISA 指令 dispatch 包装
- EmitterState 状态机（所有 ping-pong 寄存器管理）
- OffsetGenerator 融合 Pass
- Activation 融合 Pass
- PostPass：依赖分析 + 虚拟寄存器分配（src4 quirk 保留）
- load_next 多帧调度
- **FSRCNN golden 零差异对齐（1273/1274 条，PERFECT）**
- UNet 50 层结构对齐（QuantLoader 编号 1–28 验证）

### 待完成（P1）

- `bas_addr` 自动推导：DataLoader / DataStorer 的地址从硬件内存布局规则自动计算
- UNet golden 全字段对齐

### 待完成（P2）

- 完整 UNet→FSRCNN 串联流水线验证（PipelineConfig 中已预留 `inter_layer_transnum` 等参数）
- 更多模型的通用性验证
- 常量折叠可选开启（`fold_constant` 配置项已预留）

---

## 附录：核心设计约束速查

| 约束 | 描述 | 违反后果 |
|------|------|----------|
| tile_h = 32 | 硬件固定，所有模板必须遵从 | 指令数错误（1427 → 1273 案例） |
| DL/WL 共享 line_buffer_idx | toggle 只在 WL 之后 | 两者读到不同 Line Buffer slot |
| QL 和 DS 同层用相同 quant_config_idx | 层末统一 toggle 一次 | DataStorer 反量化参数错误（116 条差异案例） |
| offset_gen 不切换 feature_buf | 输出写 offset_reg，不占 a/b | 下游 dconv 读到错误 feature 数据 |
| QuantLoader layer_idx 1-based | 专用 conv_layer_counter | 编号错位 |
| src4 = src_code[2] | 移植自 golden 的 quirk | PostPass 寄存器分配与 golden 不符 |
| id() 不稳定 | 用 TVM __hash__ 而非 Python id() | DAG 遍历指数膨胀超时 |
