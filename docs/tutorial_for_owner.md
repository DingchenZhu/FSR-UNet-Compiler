# SDSR 编译器项目速通教程

> 写给项目负责人的内部教程。读者对项目有宏观了解，但对代码细节和编译器概念不够熟悉。明天需要向他人介绍这个项目——读完这份教程，你应该能自信地回答"代码是怎么写的、为什么这样设计、硬件是什么样的、踩了哪些坑"。

---

## 0. 整体一句话

> 我们写了一个编译器，能把 PyTorch / ONNX 格式的神经网络模型（FSRCNN 超分辨率模型），自动翻译成 SDSR 硬件加速器能执行的指令序列。

就像 GCC 把 C 代码变成 x86 机器码一样，我们的编译器把「神经网络计算图」变成「SDSR 硬件的 7 类指令」。

---

## 1. 为什么要做这个项目？（2 分钟能讲清楚的版本）

**问题背景**

SDSR 是一块自研的 CNN 硬件加速器芯片（复旦 IA&C Lab 设计），专门用来做超分辨率（Super-Resolution, SR）推理。它的 MAC 阵列每周期能并行处理 128 个输出，能效比 GPU 高很多。

但是，这颗芯片不能「直接跑」神经网络模型。工程师必须手工写指令序列——告诉硬件"现在从内存的哪个地址加载多少行数据、用哪组权重、往哪个 buffer 写结果"。这个过程原来是完全手写的，代码叫 `sd_sr_codegen.py`，几千行，改一个模型就要重写。

**我们的方案**

用 TVM（一个开源的神经网络编译器框架）作为前端，自动解析模型，然后通过我们自己写的 4 个阶段（Frontend → LayerDesc → Tiling → Emitter）自动生成指令序列。

验证标准是：和手写的黄金参考 `sd_sr_codegen.py` 生成的指令序列逐条对比——在 FSRCNN 模型上，我们的编译器能生成 1273 条完全一致的指令（`load_next=False` 模式）。

---

## 2. 硬件长什么样？（先理解硬件，才能理解编译器）

理解编译器设计的关键，是先理解编译器在为「什么样的硬件」服务。

### 2.1 SDSR 加速器的计算单元

核心是一个 **MAC（乘累加）阵列**，每个时钟周期可以并行处理 128 个乘累加运算。

| 规格 | 值 |
|------|----|
| 核心频率 | 200 MHz |
| MAC 并行宽度 | 128 路（每周期 128 个输出特征点） |
| 权重精度 | Sint 8bit（有符号 8 位整数） |
| 激活精度 | Uint 10bit（无符号 10 位整数） |
| 累加寄存器（ACC Reg） | 2048 个 28bit 寄存器（双 ping-pong） |
| 帧率 | 107 fps（4× SR，1080P 输入） |

类比理解：把它想象成一个「算术流水线工厂」——传送带一直走，每次送来 128 个像素点和对应权重，工厂同时完成 128 个乘法再累加进结果寄存器。

### 2.2 片上存储：每种 buffer 是干什么的

硬件有一套分层存储系统，从外到内分别是：

```
外部 DDR 内存
    ↓ OffchipDataLoader（大搬运工）
Input Buffer（片上 SRAM，存整张特征图）
    ↓ DataLoader（小搬运工）
Line Buffer（行缓冲区，最多 6 行，直接喂给 MAC）
    ↓
MAC 阵列（做乘累加）
    ↓
ACC Reg（累加寄存器，存中间结果）
    ↓ DataStorer（量化 + 写回）
Input Buffer（量化后的特征图，ping-pong 两个槽：buffer a 和 buffer b）
    ↓ OffchipDataStorer（把最终结果写回 DDR）
外部 DDR 内存
```

**每个存储层的类比**

- **DDR**：仓库，东西多但取货慢
- **Input Buffer**：车间备料区，存一批要处理的原料
- **Line Buffer**：操作台，一次只摆几行，工人（MAC）直接从这里拿
- **ACC Reg**：草稿纸，算到一半先记在这里
- **buffer a / buffer b**：两个工位，一个在计算时另一个在装卸，交替使用（这就是「ping-pong」）

另外还有两个特殊存储：
- **Weight SRAM**：存神经网络权重，由 WeightLoader 加载到 MAC
- **Quant Reg**：存量化参数（scale/bias），由 QuantLoader 加载，DataStorer 在量化时用
- **Offset Reg**：存可变形卷积的偏移量，由 OffsetLoader 加载

### 2.3 7 类指令：硬件能「听懂」什么

硬件只认 7 种指令，每条指令告诉对应的功能单元做一件事：

| 指令 | 功能 | 类比 |
|------|------|------|
| **OffchipDataLoader** | DDR → Input Buffer，大批量 DMA 搬运 | 叫货车从仓库拉货 |
| **DataLoader** | Input Buffer → Line Buffer，按行切片 | 从备料区搬到操作台 |
| **WeightLoader** | 权重 → MAC 阵列，驱动乘法运算 | 拿出食谱，开始按配方烹饪 |
| **QuantLoader** | 量化参数 → Quant Reg，预加载 | 提前备好调料 |
| **DataStorer** | ACC Reg → Input Buffer，量化后写回 | 成品装盘，放进冰箱 |
| **OffsetLoader** | Offset 数据 → Offset Reg，用于可变形卷积 | 拿到一张"调整坐标"的地图 |
| **OffchipDataStorer** | Input Buffer → DDR，最终结果写回 | 把成品打包送出厂 |

每条指令都是一个 Python 字典，长这样：

```python
{'code_num': [42], 'op_code': 'DataLoader', 'layer_idx': 0,
 'line_buffer_reshape': 0, 'is_padding_row': 1, 'read_mode': 0,
 'transnum': 4, 'line_buffer_idx': 0, 'src_buffer_idx': 'offchip_input_buffer',
 'bas_addr': 0, 'dependency': [5], 'dest': 3, 'src1': 2, ...}
```

### 2.4 最重要的硬件约束（编译器必须遵守的规则）

这些约束决定了编译器的很多设计决策，必须牢记：

| 约束 | 具体规则 | 为什么重要 |
|------|---------|-----------|
| **MAC 并行宽度 = 128** | tile_h 固定为 32（32×4=128） | 所有分块决策都从这里出发 |
| **Line Buffer 最多 6 行** | 3×3 卷积一次只能看 3-4 行 | DataLoader 的 transnum 上限 |
| **双 ping-pong Line Buffer** | line_buffer_idx 0/1 交替，DataLoader 和 WeightLoader 必须用同一个值 | 违反则硬件读到错误数据 |
| **双 ping-pong ACC Reg** | acc_reg_comp_idx 0/1 交替，DataStorer 之后才 toggle | 中间结果不能覆盖 |
| **双 Quant Reg 槽** | quant_config_idx 0/1 交替，每层一次 toggle | 上一层量化参数不能被新层覆盖 |
| **量化参数粒度** | 每 8 个输出通道共享一组参数 | QuantLoader.transnum = ceil(cout/8) |
| **权重地址两个槽** | weight_bas_addr[0] 给普通卷积，weight_bas_addr[1] 给 offset_gen | 两类权重不能混用地址 |

---

## 3. 编译器是什么？做了什么？

### 3.1 类比理解：从高级语言到机器码

普通程序员写 `a = b + c`，编译器把它变成 `ADD R1, R2, R3`——人能懂的语言变成 CPU 能懂的语言。

我们做的事情类似：

```
神经网络模型（PyTorch/ONNX）
        ↓ 我们的编译器
SDSR 指令序列（7 类 ISA 指令的 Python 字典列表）
```

### 3.2 我们的编译器：从模型到指令序列

输入：一个训练好的 FSRCNN 模型文件（`.onnx` 或 `.py`）

输出：一个文本文件，每行是一条 SDSR 指令（Python 字典格式），共 1273 条

用法示例：
```bash
python3 pipeline.py --model frontend/fsrcnn_loader.py --type pytorch \
    --input-shape 1 1 32 64 --output-dir output/fsrcnn/ --verbose
```

### 3.3 四个阶段的分工

```
[模型文件]
    ↓ Stage 1: Frontend（前端）
[Relay IR — TVM 的中间表示，一棵算子图]
    ↓ Stage 2: LayerDesc 提取 + Fusion Pass（融合）
[LayerDesc 列表 — 我们自己的中间表示，一个描述每层网络的结构体列表]
    ↓ Stage 3: Tiling（分块）
[TilingPlan 列表 — 每层的分块方案，含所有 ISA 参数]
    ↓ Stage 4: Emitter + PostPass（发射 + 后处理）
[指令序列 — 最终的 1273 条 SDSR 指令]
```

每个阶段都会把中间结果 dump 到 `output/` 文件夹，方便调试。

---

## 4. 代码实现：逐模块讲解

### 4.1 Frontend（入口）：怎么读模型

**文件位置**：`frontend/frontend.py`

**这个模块做什么**

把 PyTorch 或 ONNX 格式的模型文件，变成 TVM 的 Relay IR（一种抽象计算图表示）。

**为什么要有这个模块**

PyTorch 和 ONNX 是两种不同的模型存储格式，TVM 提供了统一的转换接口，我们封装一下就可以用。不这样做的话，后续所有阶段都要同时处理两种格式，非常麻烦。

**代码关键路径**

ONNX 路径（`load_onnx` 函数）：
```python
model_proto = onnx.load(model_path)          # 读 .onnx 文件
mod, params = relay.frontend.from_onnx(...)  # TVM 帮我们转成 Relay IR
```

PyTorch 路径（`load_pytorch` 函数）：
```python
model.eval()
traced = torch.jit.trace(model, example_inputs)  # TorchScript 追踪
mod, params = relay.frontend.from_pytorch(traced, ...)  # 再转 Relay IR
```

转换完之后，再跑一个 `relay.transform.InferType()`，这一步是让 TVM 推断每个节点的 tensor 形状——后续提取卷积层参数时需要知道输入/输出的尺寸。

**一个具体例子（FSRCNN）**

FSRCNN 的 PyTorch 模型定义在 `models_new_930.py`，入口在 `frontend/fsrcnn_loader.py`，它实现了约定的 `get_model()` 函数。pipeline 调用时会动态加载这个文件，拿到模型，然后用 `input=(1,1,32,64)` 的 zero tensor 做一次 JIT trace，转成 Relay IR。

关键发现：FSRCNN 里用了 PyTorch 的 `torchvision.ops.deform_conv2d`（可变形卷积），TVM 的 `from_pytorch` 会自动把它识别并转成 `nn.deformable_conv2d` 节点——不需要我们手工注册算子。

### 4.2 LayerDesc（中间表示）：为什么要抽象一层

**文件位置**：`ir/layer_desc.py`

**这个模块做什么**

遍历 Relay IR（算子图），把对编译器有意义的算子（卷积、池化等）提取出来，变成一个简洁的 `LayerDesc` 结构体列表。

**为什么要有这个模块**

Relay IR 是 TVM 通用的表示，里面有很多「噪音」算子，比如 `reshape`、`transpose`、`layout_transform` 等，这些在我们的硬件上不需要单独处理。另外，Relay IR 是一棵树（或 DAG），而我们的硬件是顺序执行的——需要把 DAG 拍成一个有序列表。

`LayerDesc` 就是为了屏蔽 Relay IR 的复杂性，给后续的 Tiling 和 Emitter 模块提供一个干净的接口。

**代码关键路径**

核心函数是 `extract_layer_descs(mod)`：
1. 调用 `_collect_calls_exec_order(fn.body, calls)` —— 递归遍历 Relay 的表达式 DAG，按执行顺序收集所有 `relay.Call` 节点
2. 遍历收集到的 Call 节点，按 `op_code` 匹配：
   - `nn.conv2d` → 调用 `_conv_like_from_call()` 创建 LayerDesc（提取 k_h, k_w, cin, cout, stride, padding 等）
   - `nn.deformable_conv2d` → 同上，但设 `deformable=True`
   - `nn.max_pool2d` / `nn.avg_pool2d` → 创建 pool LayerDesc
   - `reshape`、`transpose` 等 → 静默跳过
   - 未知算子 → 打印 WARNING

`LayerDesc` 长这样（以 FSRCNN 第一层为例）：
```python
LayerDesc(op='conv2d', idx=0, h_in=32, w_in=64, cin=1, cout=32,
          k_h=3, k_w=3, stride_h=1, stride_w=1,
          pad_top=1, pad_left=1, pad_bottom=1, pad_right=1,
          groups=1, deformable=False, activation='prelu')
```

**一个具体例子（FSRCNN）**

FSRCNN 原始有 23 个 Call 节点（conv2d × 12、deformable_conv2d × 4、avg_pool2d × 4、relu/prelu × 11 等），经过 `extract_layer_descs` 后得到 23 层，经过两个融合 Pass 之后精简到 12 层。

### 4.3 融合 Pass：为什么要「融合」，怎么融合

**文件位置**：`ir/fusion_pass.py`

FSRCNN 里有两种「需要特殊处理」的模式，如果不融合，后续会产生错误的指令：

#### Pass 1：OffsetGenerator 融合（`fuse_offset_generators`）

**问题背景**

可变形卷积（DeformableConv2d）需要先计算一个「偏移量图」（offset map），这个计算过程叫 OffsetGenerator，它的结构是：`AvgPool2d → Conv2d(8→18通道)` 然后把结果送给后面的可变形卷积。

在 Relay IR 里，这三层是独立的节点：`pool2d → conv2d(cout=18) → deformable_conv2d`。

如果不融合，`conv2d(cout=18)` 会走普通卷积路径，把结果写进 `buffer_a`，但硬件的 OffsetLoader 是从 `offset_reg` 读的——数据根本不在那里，可变形卷积的计算结果就全错了。

**融合规则**

识别到三层连续出现且满足以下条件时，把前两层合并为一个 `offset_gen` 节点：
```python
layers[i].op == 'pool2d'
layers[i+1].op == 'conv2d'
layers[i+2].op == 'deformable_conv2d'
layers[i+1].cout == 2 * layers[i+2].k_h * layers[i+2].k_w  # 18 = 2×3×3
```

融合后，`offset_gen` 这个特殊 op 会走专用的发射路径，DataStorer 的目标写成 `dest_buffer_idx='offset_reg'`，OffsetLoader 就能读到正确数据。

FSRCNN 里有 4 个 OffsetGenerator，融合后层数从 20 层降到 16 层。

#### Pass 2：激活函数融合（`fuse_activations`）

**问题背景**

Relay IR 里，`Conv2d` 和 `ReLU`/`PReLU` 是两个独立节点。但在 SDSR 硬件上，激活函数不是单独的指令——它是 DataStorer 指令里的一个字段（`acc_mode` / `store_mode` 控制量化后是否截断）。

如果不融合，编译器会为 ReLU/PReLU 发出 PseudoOp（占位符），但黄金参考里根本没有 PseudoOp，对比时就会出现多余指令。

**融合规则**

相邻的 `(conv2d / offset_gen / deformable_conv2d) + relu/prelu` 对，把激活类型写进前者的 `activation` 字段，丢掉独立的激活节点，重新排 idx。

融合后，FSRCNN 从 16 层降到 12 层，零 PseudoOp。

### 4.4 Tiling（分块）：为什么要分块，5 种模板是什么意思

**文件位置**：`tiling/tiling.py`

**这个模块做什么**

为每一层网络计算「分块方案」——告诉 Emitter 每次加载多少行、循环几次、用哪种计算模式。

**为什么要有这个模块**

硬件的 Line Buffer 只有 6 行宽——放不下整张 feature map。所以必须把 feature map 切成一块一块来处理，每次只处理一小块（tile）。Tiling 模块就是决定「怎么切」。

不这样做的话，Emitter 里就会出现大量 magic number，维护困难，而且不同层的切法不一样，必须系统化。

**核心概念**

- **tile_h = 32**：硬件每次处理 32 行（MAC 阵列宽度 128 = 32行 × 4分组）。对所有层固定为 32。
- **h_out_per_step**：每一个外层循环步骤（`cal_idx`）产出多少行输出。模板 A/B 是 2，模板 C 是 1，模板 E 是 4。
- **load_total_num**：一个宏 tile 内的外层 H 循环次数 = `tile_h / h_out_per_step`。tile_h=32、h_out_per_step=2 → load_total_num=16。
- **cin_group**：输入通道分组数，内层循环次数。cin=1 时 cin_group=1，cin=32 时 cin_group=8。
- **weight_transnum_base**：每次 WeightLoader 加载多少个权重。3×3 conv 是 9，1×1 conv 是 1。

**5 种模板（为什么需要这么多）**

不同的 `(cin, cout, k)` 组合对应不同的硬件计算模式，Tiling 参数完全不同：

| 模板 | 触发条件 | h_out_per_step | cin_group | 典型场景 |
|------|---------|----------------|-----------|---------|
| **C** | cin=1, k=3 | 1 | 1 | FSRCNN 第一层（单通道输入 3×3 conv） |
| **D** | k=1, cin≤8 | 1 | 1 | 小 cin 的 1×1 conv |
| **E** | k=1, cin>8 | 4 | 8 | FSRCNN L1（32→8 的 1×1 conv） |
| **F** | k=3, cin>8, cout≤8 | 4 | 8 | FSRCNN L11（32→4 的 3×3 conv） |
| **A/B** | 其余 3×3 conv | 2 | 按 cin 自动 | 大多数标准 conv 层 |

**一个具体例子（FSRCNN 第一层，Template C）**

FSRCNN Layer 0：`cin=1, cout=32, k=3, 输入 32×64`

走 Template C 的条件：`cin==1 and k==3` ✓

```
tile_h = 32
h_out_per_step = 1        → 每步产出 1 行
load_total_num = 32/1 = 32  → 外层循环 32 次
cin_group = 1             → 内层循环 1 次（单通道不用分组）
weight_transnum_base = 9  → 每次加载 3×3=9 个权重
line_buffer_rows = 3      → DataLoader 每次加载 3 行（3×3 核的行方向）
storer_step = 1           → 每步 DataStorer 地址 +1
quant_mode = 3, quant_transnum = cout = 32  → QuantLoader 参数
```

所以 FSRCNN Layer 0 会产生：
- 1 条 QuantLoader
- 32 次外层循环 × 1（cin_group=1）= 32 条 DataLoader + 32 条 WeightLoader + 32 条 DataStorer
- 合计 1+32+32+32 = 97 条指令

### 4.5 Emitter（发射器）：指令是怎么一条条生成的

**文件位置**：`backend/emitter.py`，ISA 定义在 `backend/isa.py`

**这个模块做什么**

把 `LayerDesc + TilingPlan` 变成具体的 ISA 指令字典。这是整个编译器最核心的部分。

**代码关键路径**

主函数是 `emit_program(layers, plans, ...)`，它：
1. 创建 `InstructionEmitter` 和 `EmitterState`（存各种地址计数器的可变状态）
2. 按层顺序调用 `em.emit_layer(L, P)`
3. 每层根据 `layer.op` 分发到不同的模板方法

**EmitterState：所有的「指针」都在这里**

```python
@dataclass
class EmitterState:
    line_buffer_idx: int = 0       # 当前用哪个 Line Buffer（0或1）
    acc_reg_idx: int = 0           # 当前用哪个 ACC 寄存器（0或1）
    quant_config_idx: int = 0      # 当前用哪个 Quant Reg 槽（0或1）
    feature_buf: str = "b"         # 下一层从哪个 buffer 读数据（"a"或"b"）
    weight_bas_addr: List[int]     # [0]: 普通卷积权重地址，[1]: offset_gen 权重地址
    quant_bas_addr: int = 0        # 量化参数在权重内存中的起始地址
    conv_layer_counter: int = 0    # 连续计数（只数卷积层，不数 pool/relu）
    last_layer_idx: int = -1       # 末层的 idx，末层 DS 写出口 buffer
```

**标准卷积的发射流程**

以 Layer 0 为例，`_emit_standard_conv` 的逻辑：

```
1. 发 QuantLoader（预加载这层的量化参数）
2. for 每个宏 W-tile（W=256 时有左右两半，W=64 时只有一块）：
       for load_idx in range(load_total_num):   # 外层 H 循环
           for cin_g in range(cin_group):        # 内层 cin 分组循环
               发 DataLoader（transnum=line_buffer_rows，src=上层的 feature_buf）
               发 WeightLoader（is_new=0 if cin_g==0 else 1）
               toggle line_buffer_idx
           发 DataStorer（写到 dest_buf，地址 storer_bas_addr）
           storer_bas_addr += storer_step
3. weight_bas_addr[0] += 这层权重的总大小
4. toggle feature_buf（下一层从 dest_buf 读）
5. toggle quant_config_idx
```

**ping-pong buffer 的分配逻辑**

这是最容易出错的地方，用一个小状态机追踪：

- `feature_buf` 初始值为 `"b"`
- Layer 0 的 DataStorer 写到 `"a"`（因为 dest = 与 feature_buf 相反）
- Layer 0 结束后 `feature_buf = "a"`
- Layer 1 的 DataLoader 读 `"a"`，DataStorer 写 `"b"`
- ……以此类推，层层交替
- 末层特殊：DataStorer 不写 `"a"/"b"`，而是写 `"fsrcnn_output_buffer"`（最终输出）

**OffsetGenerator 的发射**

`_emit_offset_gen` 比标准 conv 简单，固定结构：
```
QuantLoader
for ky in range(3):            # 3×3 kernel 的行方向，只有 3 行
    DataLoader（读 feature_buf，地址固定=64）
    WeightLoader（用 weight_bas_addr[1] 槽，与普通卷积不冲突）
    toggle line_buffer_idx
DataStorer（dest='offset_reg'，不写 a/b，也不切换 feature_buf）
toggle quant_config_idx
```

**可变形卷积的发射**

`_emit_deformable_conv` 结构：
```
QuantLoader
for cal_idx in range(load_total_num):   # H 方向外循环
    for ky in range(3):                 # kernel 行方向
        OffsetLoader（读 offset_reg，按 cal_idx×3+ky 地址）
        for ic_g in range(ic_inner):    # 输入通道内循环
            DataLoader（6行，因为双线性插值需要更多上下文）
            WeightLoader（is_bilinear_bicubic=1）
            toggle line_buffer_idx
        toggle offset_reg_idx
    DataStorer
    storer_bas_addr += storer_step
toggle feature_buf, toggle quant_config_idx
```

**一个具体例子（FSRCNN 端到端）**

FSRCNN 12 层的指令统计（`load_next=False`）：

| 指令类型 | 数量 |
|---------|------|
| OffchipDataLoader | 1（加载输入图像） |
| QuantLoader | 12（每层一条） |
| DataLoader | 524 |
| WeightLoader | 524 |
| DataStorer | 116 |
| OffsetLoader | 96（4个 dconv × 8 H步 × 3 ky） |
| OffchipDataStorer | 1（写出 SR 结果） |
| **合计** | **1273** |

### 4.6 PostPass（后处理）：依赖分析和寄存器分配是什么

**文件位置**：`backend/post_pass.py`

**这个模块做什么**

给每条指令填写 `dependency`、`dest`、`src1-4` 字段——这些字段描述了指令之间的数据依赖关系，以及虚拟寄存器分配结果。

**为什么要有这个模块**

SDSR 硬件支持乱序执行（out-of-order execution）。当一条指令在等待内存数据时，硬件可以提前执行另一条没有依赖关系的指令。`dependency` 字段就是告诉硬件「这条指令必须等哪些指令执行完才能开始」。

类比：就像项目管理里的甘特图——某些任务必须等其他任务完成才能开始，这里我们写的是各条指令的前驱关系。

**依赖分析的 7 条规则**

每种指令类型都有固定的依赖规则，以最常见的为例：

- **WeightLoader** 依赖：找最近一条 `line_buffer_idx` 相同的 DataLoader（因为它们操作同一个 Line Buffer）；找最近一条 `acc_reg_comp_idx` 相同的 DataStorer（因为要等上一轮结果写出后才能复用 ACC 寄存器）
- **DataStorer** 依赖：找最近一条 `quant_reg_load_idx` 匹配的 QuantLoader；找最近一条 `acc_reg_comp_idx` 匹配的 WeightLoader；找上一条 DataStorer（保证按序写出）
- **QuantLoader** 依赖：找最近一条 `quant_config_idx` 相同的 DataStorer（确保那个槽已经被消费完）

**虚拟寄存器分配**

硬件有 15 个虚拟寄存器（1-15 号）用于依赖追踪。`assign_dependency_registers` 函数做的事情类似编译器的活性分析（liveness analysis）——给每条指令分配一个 `dest` 寄存器号，用 LIFO 策略回收不再需要的寄存器。

特殊的「src4 quirk」：代码里 `src4 = src_code[2]`（第三个，而非第四个依赖的寄存器号），这是对原始黄金代码的忠实复制，是硬件或工具链的历史遗留行为。

---

## 5. 踩过的坑：关键 Bug 和修复

### Bug 1：TVM `id()` 不稳定导致 DAG 遍历超时

**现象**：对 FSRCNN 模型运行 `extract_layer_descs` 时，程序挂起超过几分钟，且提取的层数明显少于预期（缺少 AvgPool 层）。

**排查过程**：发现 `_collect_calls_exec_order` 里用了 Python 的 `id(expr)` 作为去重集合的 key，用来避免重复遍历同一个子图节点。看起来合理，但有问题。

**根因**：TVM 的 `ObjectRef`（Relay 表达式的基类）每次被 Python 访问时，都会产生一个**新的** Python 包装对象。即使底层指向同一个 C++ 节点，`id()` 返回的值也是不同的——因为 `id()` 返回的是 Python 对象的内存地址，而每次都是新包装对象。

结果：同一个节点被反复遍历，DAG 里有共享子图（FSRCNN 的 OffsetGenerator 结构被 4 个 dconv 共享），导致指数级爆炸。

**修复**：把 `visited = set()` 的使用方式从 `id(expr)` 改为直接用 `expr` 本身作为 key。TVM 的 `__hash__` 是基于底层 C++ 对象指针实现的（稳定），`__eq__` 等价于 `same_as()`（稳定的结构相等判断）。

```python
# 修复前（错误）：
if id(expr) in visited:
    return
visited.add(id(expr))

# 修复后（正确）：
if expr in visited:
    return
visited.add(expr)
```

**效果**：`extract_layer_descs` 从超时变为 0.016 秒，FSRCNN 正确提取 23 层。

---

### Bug 2：`line_buffer_idx` toggle 时机错误

**现象**：生成的指令中，同一对 DataLoader 和 WeightLoader 的 `line_buffer_idx` 值相反（DL=0，WL=1），而不是相同的值。依赖分析阶段因此无法匹配到正确的生产者。

**排查过程**：查看黄金代码 `sd_sr_codegen.py`，发现它用两个独立的 Manager（`DataLoaderManager` 和 `WeightLoaderManager`），各自从 0 开始，各自 toggle。两者始终同步，因为它们总是在同一时机 toggle。

我们的实现只有一个共享计数器，但错误地在 DataLoader **之后**、WeightLoader **之前**做了一次 toggle，导致 WL 比 DL 多 toggle 了一次，两者就不同步了。

**根因**：「DL 和 WL 必须共享同一个 `line_buffer_idx`」是硬件约束（它们操作同一个 Line Buffer 的同一个槽），toggle 只应该发生在一对 DL+WL 都完成之后。

**修复**：把 toggle 语句从 DL 和 WL 之间移到 WL 之后。

```python
# 修复前（错误）：
isa.DataLoader.dispatch(line_buffer_idx=st.line_buffer_idx, ...)
st.line_buffer_idx = 1 - st.line_buffer_idx   # ← 错误位置！
isa.WeightLoader.dispatch(line_buffer_idx=st.line_buffer_idx, ...)  # 拿到的是 toggle 后的值

# 修复后（正确）：
isa.DataLoader.dispatch(line_buffer_idx=st.line_buffer_idx, ...)
isa.WeightLoader.dispatch(line_buffer_idx=st.line_buffer_idx, ...)  # 同一个值
st.line_buffer_idx = 1 - st.line_buffer_idx   # ← 正确位置：DL+WL 都完成后
```

**效果**：依赖分析中 WeightLoader → DataLoader 的依赖边全部正确建立。

---

### Bug 3：QuantLoader `layer_idx` 编号错误（跳号）

**现象**：生成的 QuantLoader 指令里 `layer_idx` 是 1、3、5、7……（跳号），而黄金参考是 1、2、3、4……（连续）。

**排查过程**：发现代码里用 `layer.idx + 1` 作为 QuantLoader 的 `layer_idx`。`layer.idx` 是层在列表中的序号（包含 pool2d、relu 等非卷积层），而黄金只对 conv/dconv 层连续编号。

**根因**：QuantLoader 只在卷积类层（conv2d、deformable_conv2d、offset_gen）时发出，但编号用的是包含所有层的全局序号，自然会有跳号。

**修复**：在 `EmitterState` 里增加一个专用计数器 `conv_layer_counter`，只在遇到卷积类层时递增，QuantLoader 用这个计数器编号。

```python
# EmitterState 里
conv_layer_counter: int = 0  # 只数 conv/dconv/offset_gen

# emit_layer 里
if layer.op in ("conv2d", "deformable_conv2d", "offset_gen"):
    self.state.conv_layer_counter += 1
    # 发 QuantLoader 时用 conv_layer_counter
    self.emit_quant_loader(layer_idx=self.state.conv_layer_counter, ...)
```

**效果**：FSRCNN 12 个 QuantLoader 的 `layer_idx` 从 1 到 12 连续，UNet 28 个从 1 到 28 连续。

---

### Bug 4：`tile_h` 的计算理解错误（1427 → 1273 条指令）

**现象**：早期版本生成了约 1427 条指令，而黄金是 1273 条——多了约 12%。

**排查过程**：分层对比发现，普通 conv 层的 DataLoader 数量偏多，如 Layer 0 应该有 32 条 DL 但生成了 64 条。

**根因**：最初对 `tile_h`（硬件空间分块高度）的理解有偏差。正确理解是：`tile_h = 32` 是固定的硬件参数，它不随输入 feature map 的高度变化。`load_total_num = tile_h / h_out_per_step = 32/1 = 32`（模板 C 时），不是 `h_in / h_out_per_step`。

另一个相关问题是 `cin_group` 循环最初没有实现（内层循环缺失），后来加上后指令数进一步对齐。

**修复**：把 `tile_h` 在 Tiling 代码里固定为 32，`load_total_num` 计算公式明确为 `tile_h // h_out_per_step`。

**效果**：FSRCNN 指令数从 1427 降到 1273，与黄金完全一致。

---

### Bug 5：OffsetGenerator 融合 Pass 缺失

**现象**：FSRCNN 中 4 个 dconv 层的输出数值不正确（逻辑错误，难以直接验证，但通过指令比对发现：DataStorer 没有一条的 `dest_buffer_idx='offset_reg'`，而黄金里有 4 条）。

**排查过程**：查看 Emitter 对 conv2d 的处理，发现 `cout=18` 的 conv 层被当成普通 conv 处理，DataStorer 写到了 `buffer_a`，但硬件 OffsetLoader 是从 `offset_reg` 读的。

**根因**：FSRCNN 里 OffsetGenerator（AvgPool + Conv，用于生成可变形卷积偏移量）在 Relay IR 里是 pool2d + conv2d 两个节点，没有被识别为一个整体。

**修复**：在 `ir/fusion_pass.py` 里新增 `fuse_offset_generators` Pass，识别 `pool2d + conv2d(cout=2×k×k) + deformable_conv2d` 三层模式，把前两层合并为 `op='offset_gen'`，触发专用发射路径，DataStorer 目标改为 `offset_reg`。

在 `pipeline.py` 里，`extract_layer_descs` 之后立即调用这个 Pass：
```python
layers = extract_layer_descs(mod)
layers = fuse_offset_generators(layers)   # 新增这一行
```

**效果**：融合前 offset_gen 相关层发出 0 条 `dest=offset_reg` DataStorer，融合后正确发出 4 条。FSRCNN 层数从 20 降到 16（FSRCNN 有 4 个 OffsetGenerator）。

---

### Bug 6：`quant_config_idx` 相位相反（导致 116 条指令有差异）

**现象**：某阶段对比时发现，所有 DataStorer 的 `quant_config_idx` 字段与黄金相反（我们是 0 的地方黄金是 1，反之亦然），影响约 116 条 DataStorer。

**排查过程**：`quant_config_idx` 是一个双 buffer 槽的索引，每层 emit 完一次后 toggle（0→1→0→…）。Emitter 里有 `st.quant_config_idx = 1 - st.quant_config_idx`。

排查初始值发现：我们 `EmitterState.quant_config_idx` 初始值为 0，但黄金的初始状态是 1（即第一层用槽 1，第二层用槽 0）。同时，`feature_buf` 初始值也需要是 `"b"`（不是 `"a"`），才能让 layer-0 的 DataStorer 写到 `"a"`，与黄金一致。

**修复**：`EmitterState.feature_buf` 初始值从 `"a"` 改为 `"b"`；调整 QuantLoader 和 DataStorer 的 `quant_config_idx` 初始值与 toggle 时机，与黄金对齐。

**效果**：DataStorer 的 `quant_config_idx` 字段全部对齐，116 条差异消除。

---

## 6. 验证：怎么知道我们的输出是对的

### 验证标准

黄金参考是手工写的 `sd_sr_codegen.py`（`sr_inst()` 函数）生成的指令序列，存放在 `output/sr_inst_golden.txt`（1273 行，`load_next=False` 模式）。

### 验证方式

`pipeline.py` 内置了 `diff_with_golden` 函数：

```bash
python3 pipeline.py --model frontend/fsrcnn_loader.py --type pytorch \
    --input-shape 1 1 32 64 --output-dir output/fsrcnn/ \
    --golden /path/to/sr_inst_golden.txt
```

输出：
```
GOLDEN MATCH: output matches golden exactly
```

### 最终验证结果（2026-04-23）

| 模式 | 总指令数 | 与黄金比较 |
|------|---------|-----------|
| `load_next=False` | **1,273** | **完全一致，0 差异** ✓ |
| `load_next=True` | **1,274** | **完全一致，0 差异** ✓ |

按指令类型分类：

| 指令类型 | 我们 | 黄金 | 状态 |
|---------|------|------|------|
| QuantLoader | 12 | 12 | ✅ |
| DataLoader | 524 | 524 | ✅ |
| WeightLoader | 524 | 524 | ✅ |
| DataStorer | 116 | 116 | ✅ |
| OffsetLoader | 96 | 96 | ✅ |
| OffchipDataStorer | 1 | 1 | ✅ |

---

## 7. 明天介绍时的重点和常见问题参考

### 推荐介绍顺序

1. 先讲硬件（2 分钟）：这块芯片是做超分辨率的，有 MAC 阵列 + 7 类指令 + 多级 buffer
2. 再讲问题（1 分钟）：手工写指令太累，换模型就要重写几千行代码
3. 再讲方案（2 分钟）：用 TVM 前端 + 4 阶段编译器自动生成
4. 最后展示结果（1 分钟）：FSRCNN 1273 条指令，与黄金完全一致

### 常见问题及回答思路

**Q1：你们的编译器和 TVM 官方编译器有什么区别？**

TVM 官方的目标是通用 GPU/CPU，会做算子融合、自动调优（AutoTVM/Ansor）、代码生成（C++/CUDA）。我们的编译器只用了 TVM 的**前端**部分（模型解析 + Relay IR + InferType），后端完全是针对 SDSR 硬件自己写的。因为 SDSR 的指令集非常特殊（7 类手工指令，不是通用 ISA），TVM 的后端没法用。

**Q2：可变形卷积（DeformableConv2d）是怎么支持的？**

可变形卷积比普通卷积复杂得多，需要先算一个偏移量图（OffsetGenerator：AvgPool + Conv），然后用偏移量做双线性插值采样（硬件的 OffsetLoader + Bilinear 单元）。我们实现了一个专门的融合 Pass（`fuse_offset_generators`）识别这个子图，把 pool+conv 融合为 `offset_gen` op，走专用发射路径写到 `offset_reg`，再由 OffsetLoader 读出驱动双线性插值。整个过程不经过 TVM 的通用 TE schedule。

**Q3：为什么不直接用 TVM 生成代码，非要自己写 Emitter？**

SDSR 的硬件不是冯·诺依曼架构，没有通用的寄存器文件和 load/store 指令集——它是一个专用数据流处理器，用 7 类指令直接控制每个功能单元（DataLoader、MAC 阵列、DataStorer 等）。TVM 没有针对这种架构的后端，自动化代码生成 pass 也不适用。

**Q4：你们的 1273 条指令，精度是怎么保证的？**

两个层次：(1) 指令序列层面——和手工黄金参考逐条对比，字段完全一致；(2) 数值计算层面——SDSR 硬件使用 int8 权重 + int10 激活量化，量化参数（scale, zero_point）来自 QAT（量化感知训练），这部分我们目前假设已有正确的量化配置，直接读入使用。

**Q5：这套编译器能支持其他模型吗（比如 UNet）？**

架构上是支持的。Frontend 已经支持 ONNX（`load_onnx`）和 PyTorch（`load_pytorch`）双入口。LayerDesc 提取和融合 Pass 是通用的。Tiling 的 5 种模板覆盖了常见的 3×3 和 1×1 卷积组合。UNet 没有可变形卷积，反而更简单。理论上运行 `python3 pipeline.py --model unet.onnx --type onnx ...` 就能出结果。完整的 UNet 黄金对比验证是下一步工作。

**Q6：编译器生成的指令能直接在硬件上跑吗？**

目前生成的是「伪指令」（pseudo-instruction），格式是 Python 字典，与黄金 `sd_sr_codegen.py` 的输出格式完全一致。后续还需要一个「编码器」（encoder）把字典格式编码成 SDSR 20-bit ISA 的二进制比特流，才能灌入硬件。这个编码器在 `vis_compiler/emit` 目录下已有参考实现。

**Q7：踩过的最麻烦的 bug 是哪个？**

TVM `id()` 不稳定是最隐蔽的。程序没有报错，只是超时——而且你很难想到"Python `id()` 对同一个对象竟然返回不同值"这种问题。找到根因（TVM ObjectRef 每次访问都创建新的 Python 包装对象）之后修复只需要一行，但找根因花了不少时间。另一个难的是 `line_buffer_idx` 的 toggle 时机——这个必须读懂硬件 Line Buffer 的时序逻辑，才能理解为什么 DataLoader 和 WeightLoader 必须用同一个值、toggle 只能在 WL 之后做。

**Q8：为什么 Layer 0 的 DataLoader 从 `offchip_input_buffer` 读，而其他层从 `feature_buf`（buffer a 或 b）读？**

第 0 层是网络的输入层，它的输入来自 DDR 内存（通过 OffchipDataLoader 提前搬运进来的原始图像数据）。从第 1 层开始，每一层的输入都是上一层的输出（存在 ping-pong buffer a 或 b 里的中间特征图）。这个区别在 Emitter 里用一行判断处理：`src_buffer_idx = "offchip_input_buffer" if layer.idx == 0 else st.feature_buf`。

---

*文档最后更新：2026-04-24。对应代码状态：FSRCNN 1273/1274 条指令与黄金完全对齐。*
