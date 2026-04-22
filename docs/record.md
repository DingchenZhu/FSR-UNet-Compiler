# TVM 编译器前端设计项目工作日志

> 项目：面向卷积神经网络硬件加速器的 TVM 编译器前端设计与实现
> 目标网络：FSRCNN（Fast Super-Resolution Convolutional Neural Network）及其变体
> 目标硬件：自研 CNN 专用加速器（含 MAC 阵列、line buffer、可变形卷积硬件支持）

---

## 2026-04-22  项目初始化

### 工作目录创建

在 `/home/scratch.hansz_coreai/design/tvm-design/` 下建立以下子目录结构：

```
tvm-design/
├── frontend/    # TVM 前端：模型导入、Relay IR 构建、前端 Pass
├── ir/          # IR 定义与扩展：自定义算子注册、Relay/Relax 扩展
├── tiling/      # Tiling 决策层：空间/通道分块策略、地址计算
├── backend/     # 后端代码生成：指令模板、调度、微指令流生成
├── output/      # 编译输出：微指令序列、依赖图、虚拟寄存器分配结果
├── docs/        # 设计文档与论文写作
└── tests/       # 测试：单算子、端到端编译验证
```

### 探索既有脚手架：tvm-tiling / vis_compiler 流水线

阅读 `/home/scratch.hansz_coreai/design/tvm-tiling/` 下的现有代码，理解手写 codegen 全貌。

三个核心文件职责：
- `instruction.py`：ISA 定义 + 指令容器，通用工具库
- `sd_codegen.py`：UNet (stable diffusion) 手写 codegen，模型/分辨率/tiling 写死
- `sd_sr_codegen.py`：FSRCNN + offset pipeline 手写 codegen

### 识别黄金参考输出格式：7 类 ISA 指令

1. **OffchipDataLoader**：DDR → 片上 buffer
2. **DataLoader**：片上 input buffer → line_buffer（含重排、padding）
3. **WeightLoader**：权重 → MAC 阵列（k=1/3/bicubic，控制并行与 ic 累加）
4. **OffsetLoader**：offset → offset_reg（用于可变形卷积）
5. **QuantLoader**：量化参数 → quant_reg
6. **DataStorer**：acc_reg → input buffer（含 pooling、pixelshuffle、量化）
7. **OffchipDataStorer**：片上输出 buffer → DDR

依赖图：7 类指令存在严格数据依赖（生产者-消费者）；硬件使用双路 line_buffer / acc_reg / quant_reg，需做虚拟寄存器分配（0/1 切换）。

### 关键挑战：DeformableConv2d 作为一等公民硬件算子

- FSRCNN mid_part_1 / mid_part_2 均使用 DeformableConv2d
- 每个 DeformableConv2d 含 OffsetGenerator（AvgPool2d + Conv2d + repeat_interleave + pad）+ dconv（双线性插值卷积）
- 挑战：标准 TVM 会将其展开，无法识别硬件映射语义
- 设计决策：前端必须将整体 DeformableConv2d 注册为自定义一等公民算子，保留 offset pipeline 语义，以正确映射到 OffsetLoader + WeightLoader(is_bilinear_bicubic=1) 指令

### 两类输入格式

| 格式    | 代表模型                        | 导入接口                              |
|---------|---------------------------------|---------------------------------------|
| ONNX    | USR_Net、UNet                   | relay.frontend.from_onnx()            |
| PyTorch | FSRCNN (models_new_930.py) 系列 | relay.frontend.from_pytorch() / trace |

---

---

## 2026-04-22  Phase 1 编译器框架实现完成

### 完成项

| 文件 | 说明 |
|------|------|
| `backend/isa.py` | 7类ISA指令包装器，golden格式兼容（is_skip=2, is_offset=0等默认字段） |
| `ir/layer_desc.py` | Relay IR → LayerDesc，支持 conv2d/deformable_conv2d/pool2d/relu |
| `tiling/tiling.py` | TilingPlan生成，含 acc_mode/store_mode/quant_transnum 字段修正 |
| `backend/emitter.py` | 标准conv Template A + deformable conv OffsetLoader路径 |
| `backend/post_pass.py` | 依赖分析7条规则 + 虚拟寄存器分配（src4 quirk已保留） |
| `frontend/frontend.py` | ONNX (from_onnx) + PyTorch (jit.trace + from_pytorch) 双入口 |
| `pipeline.py` | 端到端编排，各阶段dump到 output/，含golden diff功能 |
| `docs/compiler_roadmap.md` | 实现路线图，含数据结构定义和DeformableConv2d完整路径 |
| `docs/architecture_recommendations.md` | 编译器专家架构建议，含P0级修正项 |
| `docs/paper_background.md` | 论文背景三部分（约3000字，含19条真实文献） |

### 专家P0级修正（已在实现中体现）

- `TilingPlan` 新增 `acc_mode`, `store_mode`, `quant_mode`, `quant_transnum` 字段
- `line_buffer_reshape` 对 w_in≤128 的层设为1，而非全局0
- `weight_transnum_base` deformable路径固定为12（bilinear 3×3）
- `src4` quirk 保留（`src_code[2]` 非 `src_code[3]`）

---

## 2026-04-22  Phase 2 — load_next 调度实现完成

### 完成项

| 文件 | 变更 | 说明 |
|------|------|------|
| `backend/emitter.py` | 新增 `_emit_preamble()` | 5条 DDR 预加载指令（quant×2 + weight×3，is_first=True） |
| `backend/emitter.py` | `emit_program()` 新参数 | `is_first`, `load_next`, `image_transnum`, `inter_layer_transnum`, `inter_layer_bas_addr` |
| `backend/emitter.py` | 修复 QuantLoader layer_idx | 改为 1-based（layer.idx+1），与 golden 对齐 |
| `backend/emitter.py` | 更新模块文档注释 | 修正 line_buffer_idx 不变式（去掉错误的"POST-toggle"说明） |
| `backend/isa.py` | OffchipDataLoader.transnum | 改为 `Any`，支持 `'unet_total'` 符号值 |
| `pipeline.py` | PipelineConfig 新字段 | `is_first`, `load_next`, `image_transnum`, `inter_layer_transnum`, `inter_layer_bas_addr` |

### load_next 调度结构（与 sd_inst 对齐）

```
[0-4]  OffchipDataLoader × 5   preamble（is_first=True）
[5]    OffchipDataLoader        image load（transnum=576, lm=0, src=0）
[6]    QuantLoader              layer_idx=1（1-based）
[7-438] DataLoader/WeightLoader/DataStorer  layer 0 的 tile loop
[439]  OffchipDataLoader        load_next（transnum=576, lm=0）若 load_next=True
[440]  OffchipDataLoader        inter_layer（transnum=64, lm=1）若设置
[441]  QuantLoader              layer_idx=2（1-based）
[442+] layer 1 tile loop...
```

### 验证结果

USR_Net.onnx + is_first=True + load_next=True：
- 生成 7305 条指令（较前 7298 多 7 条 = 5 preamble + 1 image load + 1 load_next）
- 前6条格式与 golden [0-5] 结构完全一致
- QuantLoader(layer_idx=1) 与 golden [6] 一致

### 已知遗留问题

| 问题 | 状态 | 说明 |
|------|------|------|
| deformable_conv 的 line_buffer_idx | 待修复 | 在 DataLoader 和 WeightLoader 之间仍有一次 toggle，与 standard conv 的修复方式相同 |
| QuantLoader layer_idx 不连续 | 次要 | relu/pool层跳过 QuantLoader 导致 idx 不连续（golden 使用连续编号） |
| is_new 字段版本差异 | 次要 | golden is_new=1，当前 is_new=1 已对齐 |
| Golden 中 code_num=[5] 重复 | 已分析 | golden 特有 quirk：load_next OffchipDataLoader 与 image load 拥有相同 code_num，系调度标注而非 bug |

---

## 2026-04-22  Phase 3 — Bug修复与端到端验证

### Bug修复

#### P0：deformable_conv `line_buffer_idx` Toggle 错误（`backend/emitter.py`）

**问题**：`_emit_deformable_conv` 中 DataLoader 和 WeightLoader 之间存在一次多余 toggle，导致两者收到相反的 `line_buffer_idx` 值（DL=0 vs WL=1）。

**根因**：golden `sd_sr_codegen.py` 使用两个独立的 manager，各自从0开始并独立 toggle，因此始终同步。我们用单一共享计数器模拟，但误在 DL 之后、WL 之前 toggle 了一次。

**修复**：移除 DL 与 WL 之间的 toggle，单次 toggle 保留在 WL 之后。Standard conv 路径同理。

#### P1：QuantLoader `layer_idx` 不连续（`backend/emitter.py`）

**问题**：使用 `layer.idx + 1` 作为 QuantLoader 的 `layer_idx`，导致 prelu/pool 层的 idx 被计入，出现跳号（如1→3→5）。Golden 使用对 conv/deformable_conv 层的连续1-based编号。

**修复**：在 `EmitterState` 增加 `conv_layer_counter: int = 0`，仅在 `emit_layer` 遇到 `conv2d` / `deformable_conv2d` / `offset_gen` 时递增，QuantLoader 使用该计数器。

**验证**：FSRCNN 12个 QuantLoader → layer_idx=1…12，UNet 28个 QuantLoader → layer_idx=1…28，均连续。

#### 关键发现：TVM ObjectRef `id()` 不稳定（`ir/layer_desc.py`）

**问题**：`_collect_calls_exec_order` 使用 `id(expr)` 作为去重键，导致 FSRCNN DAG 指数级重复遍历（超时）且漏提取 `nn.avg_pool2d` 节点。

**根因**：TVM ObjectRef 每次访问同一底层 C++ 节点时会产生新的 Python 包装对象，`id()` 因此不稳定。

**修复**：改用 `expr in visited` / `visited.add(expr)`，依赖 TVM 稳定的 `__hash__`（基于 C++ 对象指针）和 `__eq__`（即 `same_as()`）。

**效果**：`extract_layer_descs` 从超时变为 0.016s，FSRCNN 正确提取 23 层（含4个 `nn.avg_pool2d` 和4个 `nn.deformable_conv2d`）。

### FSRCNN 端到端打通

新增 `frontend/fsrcnn_loader.py`，封装 PyTorch 模型加载入口（`get_model()`），将 tvm-tiling/references 加入 sys.path 后导入 `models_new_930.FSRCNN`。

**关键发现**：`torchvision.ops.deform_conv2d` 被 TVM `relay.frontend.from_pytorch` 自动转换为 `nn.deformable_conv2d`，无需自定义算子注册。

运行命令：
```bash
python3 pipeline.py --model frontend/fsrcnn_loader.py --type pytorch \
    --input-shape 1 1 36 64 --output-dir output/fsrcnn/ --verbose
```

验证结果（融合前）：23层，864条指令
```
OffchipDataLoader:1, QuantLoader:12, DataLoader:304, WeightLoader:304,
DataStorer:124, OffsetLoader:108, PseudoOp:11
```

---

## 2026-04-22  Phase 4 — OffsetGenerator 子图识别与融合 Pass

### 背景与动机

`OffsetGenerator`（`models_new_930.py`）定义为 `AvgPool2d(4) + Conv2d(8→18, 3×3)`，经 TVM Relay 提取后在 LayerDesc 列表中变为连续三层：

```
pool2d (k=4×4, cin=8)  →  conv2d (cout=18, h=9×16)  →  deformable_conv2d
```

**融合前的问题**：
- `pool2d` 发出 PseudoOp（硬件忽略），AvgPool 操作语义丢失
- `conv2d(cout=18)` 走标准 conv 路径，将结果写入 `buffer a`（目标错误）
- `OffsetLoader` 读取的 `offset_reg` 是脏数据，deformable conv 计算结果不正确

**融合后的正确行为**：
- offset_gen 走专用模板，DataStorer 写入 `dest_buffer_idx='offset_reg'`
- OffsetLoader 读到正确的18通道偏移图
- 硬件数据流与 golden `sd_sr_codegen.py` 的"offset生成"段完全对应

### 实现

#### 新增文件：`ir/fusion_pass.py`

识别规则（三层连续）：
```python
layers[i].op   == 'pool2d'
layers[i+1].op == 'conv2d' and layers[i+1].cout == 18
layers[i+2].op == 'deformable_conv2d'
```

融合结果：pool2d + conv2d → 单个 `op='offset_gen'` LayerDesc（保留 conv2d 的空间/通道参数，`extra={'pool_stride': 4}`）；层号重新顺序编号。

在 `pipeline.py` Stage 2 中插入：
```python
layers = extract_layer_descs(mod)
layers = fuse_offset_generators(layers)   # ← 新增
```

#### 修改：`tiling/tiling.py`

新增 `offset_gen` 分支，所有参数由 golden 固定（不依赖运行时计算）：

| 参数 | 值 | 说明 |
|------|----|------|
| `quant_mode` | 2 | 区别于标准conv的0 |
| `quant_transnum` | 16 | QuantLoader transnum |
| `weight_transnum_base` | 24 | 每个 ky 步长的 WeightLoader transnum |
| `weight_parall_mode` | 1 | MAC 阵列下半使用 |
| `ky_outer` | 3 | 3×3 kernel 的行循环 |
| `line_buffer_reshape` | 2 | offset_gen 专用 reshape 模式 |
| `read_mode` | 1 | 区别于标准conv的0 |
| `data_bas_addr` | 64 | buffer b 中 pool 输出的固定起始地址（=32×2） |
| `acc_mode` | 1 | DataStorer 累加模式 |
| `store_mode` | 1 | DataStorer 写出模式 |

同时在 `TilingPlan` dataclass 中新增 `data_bas_addr: int = 0` 字段。

#### 修改：`backend/emitter.py`

新增 `_emit_offset_gen()` 方法，匹配 `sd_sr_codegen.py` 的"offset生成"段：

```
QuantLoader(quant_mode=2, transnum=16)
for ky in range(3):
    DataLoader(src_buffer_idx='b', bas_addr=64, line_buffer_reshape=2, read_mode=1)
    WeightLoader(bas_addr=weight_bas_addr[1] + ky×24, is_bilinear_bicubic=0)
    toggle line_buffer_idx
DataStorer(dest_buffer_idx='offset_reg', acc_mode=1, store_mode=1, stride=0)
weight_bas_addr[1] += 24×3
```

关键差异（与标准conv对比）：
- 读取 `weight_bas_addr[1]`（slot 1），不与 conv/dconv 的 slot 0 冲突
- `acc_reg_idx` 在 ky 循环内**不** toggle，仅 DataStorer 后 toggle 一次
- DataStorer 写出到 `offset_reg` 而非 `buffer a`

`emit_layer` 中 `offset_gen` 触发 `conv_layer_counter` 递增（因其发 QuantLoader）。

### 融合前后对比

| 指标 | 融合前 | 融合后 | 变化 |
|------|--------|--------|------|
| Layer 数 | 23 | **19** | −4 |
| 总指令数 | 864 | **840** | **−24** |
| PseudoOp | 11 | 7 | −4（pool2d 不再发 PseudoOp） |
| DataLoader | 304 | 300 | −4 |
| WeightLoader | 304 | 300 | −4 |
| DataStorer | 124 | 112 | −12 |
| DataStorer(dest=**offset_reg**) | 0 | **4** | +4 ✓ 正确语义 |
| OffsetLoader | 108 | 108 | 不变 |
| UNet（回归验证） | 4995 | 4995 | 无影响 ✓ |

### 收益总结

1. **正确性**：融合前 offset_gen conv 写入错误目标（buffer a），deformable conv 的偏移计算依赖脏数据；融合后数据流与硬件语义完全对应。

2. **硬件资源对齐**：offset_gen 独占权重槽 `weight_bas_addr[1]`，标准/deformable conv 使用槽 `[0]`，两者互不干扰，与 golden 的双槽管理（`weightloadermanager.bas_addr_cur[0/1]`）完全一致。

3. **指令精简**：offset_gen 8条指令/层（vs 标准conv 13条），4个 OffsetGenerator 节省20条；消除4条 PseudoOp；净减24条指令。

## 当前状态与后续计划

### 已完成（截至 2026-04-22）

| 文件 | 状态 |
|------|------|
| `backend/isa.py` | ✅ |
| `ir/layer_desc.py` | ✅ TVM id() 修复 |
| `ir/fusion_pass.py` | ✅ OffsetGenerator 融合 |
| `tiling/tiling.py` | ✅ offset_gen 分支 |
| `backend/emitter.py` | ✅ 含 P0/P1 修复 + offset_gen 模板 |
| `backend/post_pass.py` | ✅ |
| `frontend/frontend.py` | ✅ |
| `frontend/fsrcnn_loader.py` | ✅ |
| `pipeline.py` | ✅ 含融合 Pass 插入 |

### 待完成

- [ ] 中文学术论文实验章节（待有 golden 对比数据后展开）
