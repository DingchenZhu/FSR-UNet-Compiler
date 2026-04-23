# CNN硬件加速器架构与编译器设计深度分析

> 由 tvm-compiler-expert 产出，2026-04-22。用于论文写作参考。

---

## A. 硬件架构总览

### A.1 加速器类型定位

该加速器属于**行缓冲流式 Dataflow 架构**（Line-Buffer Streaming Dataflow），而非传统的脉动阵列（Systolic Array）或时间复用型架构。其核心计算范式为：以 feature map 的"行组"（H-tile）为基本流动单元，数据从 DRAM 流入行缓冲区（line buffer），经 MAC 阵列完成滑窗卷积，结果经量化写回片上 buffer 或 DRAM。

显著特征：
- 卷积窗口不需要把整个 feature map 搬入片上，以流水方式逐批处理
- 原生支持 DeformableConv2d 的双线性插值权重计算
- 原生 PixelShuffle/pooling 后处理单元集成于 DataStorer
- 双路乒乓寄存器（line buffer、acc_reg、quant_reg 各两份）支持流水重叠

### A.2 关键片上资源

| 资源 | 规格 | 对应 ISA 字段 |
|------|------|--------------|
| Line Buffer | 双路（0/1），每路存若干行激活数据 | `DataLoader.line_buffer_idx` |
| Accumulation Register (acc_reg) | 双路（0/1），存储 MAC 部分和 | `WeightLoader.acc_reg_comp_idx`，`DataStorer.reg_out_idx` |
| Quant Config Register | 双路（0/1），存量化 scale/zero-point | `QuantLoader.quant_reg_load_idx`，`DataStorer.quant_config_idx` |
| Offset Register | 双路（0/1），存 deformable conv 偏移图 | `OffsetLoader.offset_reg_idx`，`WeightLoader.offset_reg_idx` |
| 片上 Input Buffer A | 通用激活暂存（层间中转） | `DataStorer.dest_buffer_idx='a'` |
| 片上 Input Buffer B | offset_gen 中间结果、deformable 输入 | `DataLoader.src_buffer_idx='b'` |
| 片上 Weight Buffer | 三个槽位 [0][1][2] | `WeightLoader.bas_addr + weight_parall_mode` |
| 片上 Quant Buffer | 量化参数存储区 | `QuantLoader.bas_addr` |

### A.3 单层卷积的硬件执行模型

以 UNet Layer 0（256×144，3×3 conv，cin=4）为例：

```
阶段0（模型初始化，is_first=True）：
  OffchipDataLoader × 5  →  将权重和量化参数从 DRAM 搬入片上
                             [quant×2 (src=2) + weight×3 (src=1)]

阶段1（输入图像加载）：
  OffchipDataLoader      →  transnum=576, 当前帧图像 → offchip_input_buffer

阶段2（量化参数配置）：
  QuantLoader            →  quant_reg_load_idx=0, layer_idx=1, transnum=4

阶段3（H 方向 tile 循环，72 次，每步 2 行）：
  for load_idx in range(72):           # H/2 = 72 步
    for cin_g in range(cin_group):     # cin_group=4（Layer 1 示例）
      DataLoader    → line_buffer[lb], src='a', bas_addr=base+h_in*cin_g
      WeightLoader  → acc_reg[acc], lb_idx=lb,
                       is_new=(0 if cin_g==0 else 1),
                       bas_addr=weight_base + cin_g*transnum
      lb ^= 1                          # DL+WL 之后 toggle
    DataStorer      → acc_reg[acc] → buffer a, storer_addr += 2
    acc ^= 1                           # DataStorer 之后 toggle

阶段4（W 方向宏 tile：W=256 时切分左半 [0:128] 和右半 [128:256]）
```

**关键执行不变式**（编译器严格维护）：
- DL 和 WL 同一次调用必须使用相同的 `line_buffer_idx`，不可在两者之间 toggle
- `line_buffer_idx` toggle 发生在 WL 之后
- `acc_reg_idx` toggle 发生在 DS 之后（不在 cin 内层循环内部）
- `is_new=0` = 重置累加器（新输出通道），`is_new=1` = 继续累加（cin 方向部分和）

---

## B. ISA 七类指令语义精解

### OffchipDataLoader — DRAM→片上 DMA
触发时机：
1. **模型初始化**（is_first=True）：预加载全部权重和量化参数，后续无需再次加载
2. **每帧图像加载**：将输入 feature map 搬入 offchip_input_buffer
3. **load_next 预取**：Layer 0 计算完成后预取下一帧，实现帧级流水

### DataLoader — 片上 buffer→Line Buffer
`line_buffer_reshape`（0=标准，1=窄层，2=offset_gen）控制行缓冲区重排模式。`is_padding_row`（0~7）覆盖所有 padding 边界类型，硬件在 DataLoader 阶段处理边界，无需实际填零。

### WeightLoader — 权重 buffer→MAC 阵列/acc_reg
`kernal_size`（0=3×3，1=1×1）切换 MAC 阵列计算模式。`weight_parall_mode`（0/1/2）对应 weight buffer 三个槽。`is_bilinear_bicubic=1` 激活双线性插值权重计算路径（deformable conv 专用）。

权重复用核心：cin=1 的层（如 Layer 0），全部 72 次 WeightLoader 指向同一地址（`bas_addr_cur` 不递增），实现完美权重复用。

### QuantLoader — 量化参数→quant 寄存器
`quant_mode`（0=INT8，2=offset_gen，3/5/7=特殊混合精度）。`layer_idx` 为 **1-based 连续编号**，仅对 conv/dconv/offset_gen 计数，relu/pool 不计入。双路 ping-pong 使当前层计算与下一层参数加载重叠。

### DataStorer — acc_reg→片上 buffer（计算触发与存储边界）
DataStorer 是计算与存储的边界点：此时 WeightLoader 已完成当前 tile 全部 cin 累加，acc_reg 存有完整部分和。关键字段：
- `dest_buffer_idx='offset_reg'`：**OffsetGenerator 专用路径**，输出写入 offset 寄存器而非数据 buffer
- `is_pooling=1`：对输出做 pooling（deformable conv 路径）
- `is_pixelshuffle=1`：PixelShuffle 上采样（FSRCNN last_part）

### OffsetLoader — offset_reg 内部寻址
在每个 deformable conv 的 `ky` 迭代前，从 offset_reg 取出当前 tile 的空间采样偏移，传给随后的 WeightLoader。`bas_addr = cal_idx * ky_outer + ky`，线性递增。

### OffchipDataStorer — 片上 buffer→DRAM
在整个模型最后一层之后发出，将最终输出（如 FSRCNN 超分辨率结果）写回 DRAM。

---

## C. 五大加速机制

### C.1 空间分块（Spatial Tiling）
Line buffer 是固定容量片上 SRAM，无法容纳整张 feature map，必须切分 H 方向：

| 层类型 | W | `h_out_per_step` | `cal_total_num` |
|--------|---|-----------------|----------------|
| 标准 conv | 256 | 2 行/step | H_out // 2 |
| 标准 conv | ≤128 | 1 行/step | H_out |
| 可变形 conv | 任意 | 4 行/step | H_out // 4（+1 尾 tile） |

W 方向：W=256 切分为左半（W=128）和右半（W=128）两个宏 tile，两者之间不重发 QuantLoader。

### C.2 乒乓缓冲（Ping-Pong Buffering）
```
时间步 t：   DL → line_buffer[0]，WL ← line_buffer[0]；toggle → lb=1
时间步 t+1： DL → line_buffer[1]，WL ← line_buffer[1]；toggle → lb=0
```
`is_new=0` 时，MAC 阵列处理 buffer[0] 的数据，DMA 可同时向 buffer[1] 预取，实现计算与访存重叠。

### C.3 权重复用（Weight Reuse）
cin=1 层：72 次 WeightLoader 全部指向同一权重地址，完美复用。cin>1 层：权重在 cin 方向按 `weight_transnum_base` 递增，层间基地址按 `transnum × cin_group` 推进。

### C.4 acc_reg 片上累加（On-Chip Partial Sum）
cin_group > 1 的层，多次 WeightLoader 通过 `is_new` 字段在片上 acc_reg 中累加 cin 方向部分和，**避免 cin 方向中间结果写回 DRAM**。双路 acc_reg ping-pong 使 DataStorer 读出当前 tile 结果的同时，MAC 阵列可写入下一 tile。

### C.5 DeformableConv2d 原生支持

**阶段一（OffsetGenerator）**：
```
DataLoader(src='b') → WeightLoader(is_bilinear_bicubic=0) × 3 ky
DataStorer(dest_buffer_idx='offset_reg')   # 写入 offset 寄存器
```

**阶段二（Deformable Conv）**：
```
for cal_idx in range(H//4):
    for ky in range(3):
        OffsetLoader(offset_reg_idx, bas_addr=cal_idx*3+ky)
        for ic_g in range(2):
            DataLoader(transnum=6)   # 6 行（双线性插值需额外行）
            WeightLoader(is_bilinear_bicubic=1)
    DataStorer(is_pooling=1)
```

`is_bilinear_bicubic=1` 激活 MAC 阵列的双线性插值权重计算，空间采样和插值全部在片上硬件完成。

---

## D. 编译器设计与五个 Pass

### D.1 为何需要编译器（手写 codegen 的瓶颈）

手写 codegen 复杂度 O(N·M)（N=模型数，M=算子类型数）：
- `sd_sr_codegen.py` 约 3000+ 行，仅覆盖 UNet+FSRCNN
- 分辨率变化需全局搜索替换所有硬编码常量
- 虚拟寄存器（`line_buffer_idx` 等）靠 manager 对象手动管理，错误为静默运行时错误

TVM 编译器前端复杂度 O(N+M)：新模型只需提供模型文件，新算子只需扩展 LayerDesc 和 emitter template。

### D.2 各 Pass 设计动机

#### Pass 1：Relay IR 提取（`ir/layer_desc.py`）
提取空间维度（h_in, w_in）、通道（cin, cout）、卷积参数（k_h, stride, pad）、可变形标志、op 类型，供后续 pass 使用。

关键技术问题：TVM ObjectRef `id()` 不稳定（每次 Python 访问产生新包装对象）。修复：用 `expr in visited`（依赖 TVM 稳定的 `__hash__`），FSRCNN 提取从超时变为 16ms。

#### Pass 2：OffsetGenerator 融合（`ir/fusion_pass.py`）
识别三层模式：`pool2d → conv2d(cout=18) → deformable_conv2d`，融合为 `offset_gen` 虚拟算子。

动机：硬件对 offset_gen 有专用数据流路径（`DataStorer.dest='offset_reg'`），标准 Relay IR 无法表达。不融合则生成语义错误的指令（写错目标 buffer），是**正确性的必要条件，而非性能优化**。

收益（FSRCNN）：23 层 → 19 层；864 → 840 条指令（−2.8%）；4 条 `DataStorer(dest='offset_reg')` 正确出现。

#### Pass 3：Tiling 规划（`tiling/tiling.py`）
将层参数映射为 `TilingPlan`（`cin_group`、`h_out_per_step`、`load_total_num`、`w_macro_tiles`、`weight_transnum_base` 等），由硬件物理约束决定，无需搜索。

#### Pass 4：指令发射（`backend/emitter.py`）
`EmitterState` 维护跨层可变状态（`line_buffer_idx`、`acc_reg_idx`、`weight_bas_addr`、`conv_layer_counter` 等）。三条发射模板：标准 conv、offset_gen、deformable conv，维护全部硬件不变式。

#### Pass 5：后处理与虚拟寄存器分配（`backend/post_pass.py`）
- **依赖分析**：按 7 条规则填充 `dependency` 字段（硬件调度器使用）
- **虚拟寄存器分配**：将符号名（'a'/'b'/'offset_reg'）解析为物理寄存器编号，填充 `dest`/`src1`-`src4`
- **src4 quirk**：依赖数达到 4 时，`src4=src_code[2]`（非 src_code[3]），来自 golden parity，必须保留

---

## E. 与相关工作的比较

### E.1 vs NVDLA 编译器
- **更底层**：直接生成微指令序列（DL/WL/DS/OL 各条），而非调用硬件"层执行引擎"
- **更精确**：需维护 `line_buffer_idx`/`acc_reg_idx` 等寄存器状态机
- **更灵活**：OffsetGenerator 融合 pass 表达了 NVDLA 无法原生支持的可变形卷积路径

### E.2 vs VTA（Versatile Tensor Accelerator）
- VTA 指令集（LOAD/GEMM/STORE）不含 deformable conv / bilinear interpolation 等特化原语
- VTA 用 AutoTune 搜索 tiling，本项目 tiling 由物理约束完全决定（非搜索问题）
- VTA 无需编译器端虚拟寄存器分配（由微架构内部处理）

### E.3 vs Halide/TVM 标准调度
- `line_buffer_idx` 的有状态全局寄存器无法表达为无副作用的循环变换
- OffsetGenerator 输出写入 `offset_reg` 不符合 Halide 的生产者-消费者内存模型
- DataStorer 的 pooling/pixelshuffle 是与计算耦合的写出操作，不能分离为独立后处理算子

---

## F. 论文贡献点（5 项）

### 贡献1：OffsetGenerator 子图硬件感知融合 Pass
标准 IR 无法表达"Conv 输出写入 offset 寄存器"的硬件语义。设计模式匹配融合 pass，识别 `pool2d→conv2d(cout=18)→deformable_conv2d` 三层子图，触发专用 `dest_buffer_idx='offset_reg'` 写出路径。此为正确性的必要条件。

### 贡献2：Line Buffer 双路乒乓不变式的编译器端形式化
形式化不变式："DL 和 WL 同一调用对使用相同 `line_buffer_idx`，toggle 发生在 WL 之后"，在三条发射路径（standard conv / offset_gen / deformable conv）中一致维护。不变式违反是静默错误，只能通过 golden-diff 检出。

### 贡献3：DeformableConv2d 作为一等公民硬件原语的端到端编译
现有编译器（NVDLA、VTA、Ansor）均不支持 DeformableConv2d 为硬件原语。本编译器利用 TVM Relay 的 `nn.deformable_conv2d`，从 PyTorch 模型到 ISA 微指令序列全自动，实现双线性插值模式的硬件加速。

### 贡献4：双 Weight Buffer 槽的编译器端地址管理
`offset_gen` 层使用槽 [1]，标准/deformable conv 使用槽 [0]，两者地址独立推进互不干扰。`EmitterState.weight_bas_addr` 为三元素列表，自动管理槽选择和层间推进。

### 贡献5：TVM ObjectRef 哈希不稳定性修复及对 DAG 遍历的影响
TVM Python 绑定的非文档化行为：同一 C++ 节点每次 Python 访问产生新包装对象，`id()` 不稳定，导致含共享子图的 DAG（如 FSRCNN）指数级重复遍历。修复：改用 `expr in visited`（依赖 `__hash__`），FSRCNN 提取从超时变为 16ms。

---

## G. 系统级编译流程

```
PyTorch/ONNX 模型
    ↓ relay.frontend.from_pytorch / from_onnx
Relay IR（含 nn.deformable_conv2d 原语）
    ↓ extract_layer_descs（DAG 遍历 + TVM 类型推断）
LayerDesc 序列（23 层，FSRCNN 示例）
    ↓ fuse_offset_generators（子图融合 Pass）
融合后 LayerDesc 序列（19 层，4 个 offset_gen 虚拟算子）
    ↓ compute_tiling_plan（硬件约束驱动的 tiling 决策）
TilingPlan 序列（含 cin_group / h_out_per_step / w_macro_tiles 等）
    ↓ emit_program（三条发射模板 + EmitterState 状态机）
原始 ISA 指令流（840 条，FSRCNN 示例）
    ↓ finalize_instructions（依赖分析 + 虚拟寄存器分配）
最终微指令序列（golden 格式，含 dependency / dest / src1-4 字段）
```

手写 codegen（UNet + FSRCNN）约 3000+ 行，与模型强绑定。编译器方案中，新模型只需提供模型文件，全流程自动完成。实测验证：FSRCNN 12 个 QuantLoader 连续编号（1-12），8/12 层指令完全匹配 golden，4 个 offset_gen 和 4 个 deformable_conv 层零差异。
