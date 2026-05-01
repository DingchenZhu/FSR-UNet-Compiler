# 地址分配调研报告：SD-UNet 加速器三类地址推导

**调研日期：** 2026-04-28  
**基于文件：** `ir/addr_alloc.py`、`ir/layer_desc.py`、`backend/emitter.py`、`pipeline.py`、`references/sd_sr_codegen.py`

---

## 一、背景与问题定义

当前编译器指令数已与 golden 完全对齐（17155/17155），功能字段（is_pooling、acc_mode 等）全部正确，唯独以下三类地址字段全部置 0：

| 字段 | 位置 | 当前状态 |
|------|------|---------|
| `DataStorer.base_addr_pooling` | emitter.py L323/532/604 | 硬编码 0 |
| `DataLoader.bas_addr` (skip) | emitter.py _emit_w_macro_tile | `st.layer_input_bas_addr` 已接入但值错误 |
| `DataStorer.base_addrs_res` | emitter.py `st.storer_bas_addr` | `st.layer_output_bas_addr` 已接入但值错误 |

这三类地址的错误根源指向同一处：`ir/addr_alloc.py` 的 `_output_size_words()` 公式错误，以及 `base_addr_pooling` 完全未被 `addr_alloc` 感知。

---

## 二、三类地址逐一分析

### 2.1 DataStorer.base_addrs_res（主输出地址，per-DS 递增）

#### Golden 中的规律

| 层（golden idx） | 写入 buffer | base_addrs_res 初始值 | 每步增量 (storer_step) |
|-----------------|------------|----------------------|----------------------|
| Layer 0（c1, h=144, W-split 左） | a | 0 | 2 |
| Layer 0（c1, h=144, W-split 右） | a | 144×4 = 576 | 2 |
| Layer 1（c1_1, h=144, W-split 左） | b | 0 | 2 |
| Layer 1（c1_1, h=144, W-split 右） | b | 576 | 2 |
| Layer 2（c1_2, h=144, W-split 左，skip 保存） | a | 0 | 2 |
| Layer 2（W-split 右） | a | 576 | 2 |
| Layer 3（conv2, h=72, 单 tile） | b | 0 | 8 |
| Layer 4（conv3, h=72, skip 保存） | a | 1152 (=144×4×2) | 8 |
| Layer 5（conv4, h=36） | b | 0 | 8 |
| Layer 6（conv5, h=36, skip 保存） | a | 1728 (=1152+576) | 8 |
| Layer 8（conv7 group, h=18, L1_0） | a | 2160 (=1728+288+144) | 1 |
| Layer 8（conv7 group, h=18, L1_1） | a | 2160+18×8=2304 | 1 |

#### 推导规则

**base_addrs_res 起始值** = `layer_output_bas_addr`，即当前层输出张量在目标 buffer 中的起始字地址。这正是 `addr_alloc.py` 试图计算的量，但其 `_output_size_words()` 公式有严重错误（见 2.4 节）。

**storer_step** = `plan.storer_step`，已在 `TilingPlan` 中正确设置（Layer 0-2 为 2，Layer 3+ 为 1 或 8 等），emitter 也已在 `st.storer_bas_addr += plan.storer_step` 处正确使用。

**结论**：`base_addrs_res` 的 per-DS 递增逻辑已正确实现。唯一缺口是 `layer_output_bas_addr` 的起始值错误（来自 `addr_alloc`）。

---

### 2.2 DataStorer.base_addr_pooling（Pool 输出地址）

#### Golden 中的规律

Golden 在每个 "pool-while-store" 层的 DS 循环前初始化 `base_addr_pooling_cur`，每次 DS 步后按固定步长递增：

| 层 | pool_output_mode | base_addr_pooling 初始值 | 每步增量 |
|----|-----------------|--------------------------|---------|
| Layer 2（c1_2, h=144） | 0 | 1152 (=144×4×2) | 4（每 DS 步） |
| Layer 4（conv3, h=72） | 1 | 1728 (=1152+576) | 4（每 2 DS 步） |
| Layer 6（conv5, h=36） | 1 | 2016 (=1728+288) | 4（每 2 DS 步） |
| Layer 8（conv7 group, L1=0） | 2 | 2016 | 8（每 2 DS 步） |
| Layer 8（conv7 group, L1=1） | 2 | 2016+9×8=2088 | 8（每 2 DS 步） |

**关键注意**：pool 输出紧跟在对应主输出张量之后存放。Layer 2 的主输出（c1_for_cat）占 buffer A 的 [0, 1152)，pool 输出（c1_pool_out）紧接在 1152 处开始。

#### 推导公式

```
base_addr_pooling_start =
    sum(sizes_of_all_prior_live_skip_tensors_in_buffer_A)
  + size_of_current_layer_main_skip_tensor
= "end_addr_of_current_main_skip_tensor_in_buffer_A"
```

数值验证：
- Layer 2：0（无先前 skip）+ 1152（c1_for_cat 大小）= **1152** ✓  
- Layer 4：1152（c1_for_cat）+ 576（c3_for_cat 大小）= **1728** ✓  
- Layer 6：1152+576（c1+c3_for_cat）+ 288（c5_for_cat 大小）= **2016** ✓  
- Layer 8 L1=0：pool 在 c5_pool 所在区（c5_pool 此时已 invalid），起始 = **2016** ✓

#### Pool 输出大小（用于验证地址范围）

| 层 | pool 输出 | 大小（words） | 计算方式 |
|----|----------|-------------|---------|
| c1_pool (h=72, w=128) | c1_pool_out | 288 | 72×4 |
| c3_pool (h=36, w=64) | c3_pool_out | 144 | 36×4 |
| c5_pool (h=18, w=32) | c5_pool_out | 72 | 18×4 |
| c7_pool (h=9, 2 groups) | c7_pool_out | 144 | 9×8×2 |

**重要发现**：`base_addr_pooling` 完全未在 `addr_alloc.py` 和 `emitter.py` 中实现，当前所有 DS 指令的该字段均为 0（硬编码于 emitter.py L323/532/604）。

---

### 2.3 DataLoader.bas_addr（Skip Connection 输入地址）

#### Golden 中的规律

Decoder 层在读取 skip connection 时，DL `bas_addr` 直接指向 buffer A 中对应 skip 张量的存储位置：

| Decoder 层 | 读取的 skip 张量 | DL bas_addr |
|-----------|----------------|------------|
| Layer 13（conv12，解码 c5） | c5_for_cat | 1728 |
| Layer 15（conv14，解码 c3） | c3_for_cat | 1152 |
| Layer 17（conv16，解码 c1） | c1_for_cat | 0 |

这正是各 skip 张量在 buffer A 中的起始地址，与编码阶段写入时的 `base_addrs_res_cur` 完全对应。

#### 现有实现状态

`addr_alloc.py` 已有 Linear Scan + ILP 实现（Phase 11），`emitter.py` 的 `st.layer_input_bas_addr` 已从 `addr_map` 读取。问题在于：

1. `addr_map` 的值由 `_output_size_words()` 驱动，而该公式有严重错误
2. 解码层 skip 连接的 `bas_addr` 需要读取的是**编码层输出张量的起始地址**，addr_map 的 key 是 `layer.idx`，这在结构上是正确的

---

### 2.4 根本问题：`_output_size_words()` 公式错误

**当前公式**（`addr_alloc.py` L47-55）：

```python
def _output_size_words(layer: LayerDesc) -> int:
    h_out = max(1, layer.h_in // layer.stride_h)
    w_out = max(1, layer.w_in // layer.stride_w)
    return h_out * max(1, math.ceil(w_out * layer.cout / 64))
```

**错误分析**：

对于 Layer 2（h=144, w=256, cout=64）：
- 当前公式：`144 × ceil(256×64/64) = 144 × 256 = 36864 words`
- Golden 实际大小：`144 × 4 × 2 = 1152 words`
- 误差：**32 倍**（等于 cout/2 = 64/2）

**正确的张量大小计算方式**：

在硬件中，"一个字（word）" = 64 像素（一个 MAC 阵列列宽突发）。通道数通过 MAC 并行隐式处理，**不直接乘以**字数。实际上，每行的字数为：

```
words_per_row = ceil(w_out / 64)      # 每行 64px 为一个字，与 cout 无关
```

对于 W-split 层（2 个宏 tile），每个 tile 写 `h_out × words_per_half_row` 的区域，但两个 tile 各起始于不同偏移，整体预留范围为：

```
region_size = h_out × (w_in / 64) × 2   # 两个半宽度 tile 的总边界框
```

验证：`144 × (256/64) × 2 = 144 × 4 × 2 = 1152` ✓

对于单 tile 层，大小直接等于：
```
size = load_total_num × storer_step
```

验证（Layer 3，h=72，storer_step=8）：`72 × 8 = 576` ✓（golden c2_out=[0,576)）

**真正正确的公式需要结合 TilingPlan**，无法仅从 LayerDesc 推导。

---

## 三、现有 Linear Scan 与 Golden 的差距

| 方面 | addr_alloc.py 现状 | Golden 实际 |
|------|-------------------|------------|
| 张量大小公式 | `h_out × ceil(w×cout/64)`（错误，误差 32×） | `load_total × storer_step × num_tiles` |
| 地址空间 | 理论上覆盖 a/b buffer 各自区间 | Buffer A 作为持久 skip 仓库，B 为当前计算临时区 |
| pool 输出 | 完全未追踪 | 紧跟主输出张量后分配 |
| 分配策略 | 动态 Linear Scan | 静态累积布局（encoder 层按层顺序追加） |
| 解码层 DL 地址 | 从 `addr_map[last_feature_layer_idx]` 读 | 直接指向 buffer A 中对应 skip 张量位置 |

**结论**：现有 Linear Scan 的**架构方向正确**（live interval + 同一 buffer 内非重叠），但执行层面存在两个关键 bug：

1. 张量大小公式错误 → 产生的 addr_map 数值错误
2. pool 输出地址完全缺失 → base_addr_pooling 恒为 0

---

## 四、可行方案对比

### 方案 A：完全硬编码 Golden 值

直接将 golden 中各层的 `base_addr_pooling`、`base_addrs_res` 起始值、DL `bas_addr` 作为查找表写入 tiling.py 的 override 列表或 emitter.py 的 if-else 分支。

| 维度 | 评估 |
|------|------|
| 工作量 | 最小（~1 天）|
| 准确性 | 100%（直接抄 golden）|
| 通用性 | 零（仅适用 USR_Net_109，输入 144×256）|
| 上板速度 | 最快 |
| 风险 | 硬编码值分散，维护困难；不同输入尺寸立即失效 |

**实现路径**：
```python
# tiling.py overrides 中追加：
{
    "idx_match": [2],   # pool 层
    "pool_addr_start": 1152,
    "pool_step": 4,
},
```
或直接在 emitter.py 的 `_emit_w_macro_tile` 中加 `if layer.idx in (2,5,8,11)` 分支。

---

### 方案 B：基于层结构自动推导（扩展现有 Linear Scan）

修复 `addr_alloc.py`，使其能正确计算张量大小并追踪 pool 输出地址，然后将 pool_addr_map 传入 emitter。

#### B1：修复张量大小公式（依赖 TilingPlan）

在 `pipeline.py` 的 Stage 3.5 中，将已计算好的 TilingPlan 传给 `allocate_addresses()`，改用：

```python
def _output_size_words_from_plan(plan: TilingPlan) -> int:
    """张量在 buffer 中的最大地址范围（边界框）。"""
    num_tiles = len(plan.w_macro_tiles)
    if num_tiles == 2:
        # W-split：右 tile 起始 = h_out * words_per_full_row
        # 总范围 = right_start + load_total * storer_step
        h_out = plan.load_total_num * plan.h_out_per_step
        words_per_full_row = plan.tile_h * 4  # tile_h=2 for layer 0 -> 8 words/row? 
        # 实际：右 tile bas_hint = h_in * (w_in/64) for most layers
        right_start = plan.w_macro_tiles[1][2]  # bas_hint of right tile
        left_data = plan.load_total_num * plan.storer_step
        return right_start + left_data
    else:
        return plan.load_total_num * plan.storer_step
```

#### B2：追加 pool 输出分配

在 `allocate_addresses()` 中新增 `pool_addr_map: Dict[int, int]`，遍历 skip 层并在 main 输出区之后累积分配 pool 输出区：

```python
pool_base = main_skip_end  # main 张量的结束地址
for layer in skip_layers:
    pool_addr_map[layer.idx] = pool_base
    pool_base += pool_output_size(layer, plan)  # h_pool * pool_words_per_row
```

#### B3：扩展 emitter 接口

将 `pool_addr_map` 与 `addr_map` 并行传入 emitter，在 DS 指令中使用：

```python
base_addr_pooling = pool_addr_map.get(layer.idx, 0) + pooling_iter_offset
```

| 维度 | 评估 |
|------|------|
| 工作量 | 中等（3-5 天，主要在公式推导和测试）|
| 准确性 | 高（依赖公式正确性，需充分测试）|
| 通用性 | 高（可支持不同输入尺寸、不同深度 UNet）|
| 上板速度 | 中等 |
| 风险 | W-split 层的 bas_hint 计算需仔细核对；pool_mode 差异需分支处理 |

---

### 方案 C：TilingPlan 驱动的静态布局（最接近 Golden 逻辑）

在 `plan_all()` 之后新增一个 `compute_static_layout()` pass，按 golden 的累积分配逻辑生成完整的 buffer A 布局表，返回：

```python
@dataclass
class BufferLayout:
    addr_map: Dict[int, int]      # layer_idx -> base_addrs_res start
    pool_addr_map: Dict[int, int] # layer_idx -> base_addr_pooling start
    pool_step_map: Dict[int, int] # layer_idx -> pool_storer_step
```

Pass 逻辑：
1. 遍历 encoder 层，检测 `plan.has_pool_output == True` 的层
2. 对每个 skip 层，根据 `plan.load_total_num * plan.storer_step * num_tiles` 累积记录 main 输出区域
3. pool 输出区紧跟 main 输出区之后
4. 将整个布局表传入 emitter

这与方案 B 的区别在于：方案 B 试图修复通用 Linear Scan（保留活跃区间分析），方案 C 直接模拟 golden 的静态分配逻辑（假设已知 skip 结构）。

| 维度 | 评估 |
|------|------|
| 工作量 | 中等（3-4 天）|
| 准确性 | 最高（逻辑最接近 golden）|
| 通用性 | 中等（需假设 UNet 式 skip 结构）|
| 上板速度 | 中等 |
| 风险 | 依赖 `skip_sources` 正确传播；decoder 层的 DL bas_addr 需从 layout 反查 |

---

## 五、方案对比总结

| | 方案 A（硬编码） | 方案 B（修复 Linear Scan） | 方案 C（静态累积布局） |
|--|:-:|:-:|:-:|
| 工作量 | 1 天 | 3-5 天 | 3-4 天 |
| 准确性 | 100%（golden 值） | 高（公式正确性依赖） | 最高 |
| 通用性 | 无 | 高 | 中等 |
| 上板速度 | 最快 | 慢 | 快 |
| 测试难度 | 低 | 高 | 中 |
| 代码复杂度 | 极低 | 高（需扩展多个模块） | 中 |

---

## 六、推荐方案

### 推荐：方案 A + 方案 C 分阶段实施

**阶段一（优先，目标：最快上板）**：实施方案 A

将 golden 中四个 pool 层（idx=2,5,8,11）的 `base_addr_pooling` 起始值和 main 输出起始值作为 tiling.py override dict 中的显式字段写入，同时修复 emitter 中 `base_addr_pooling` 字段的赋值逻辑（当前硬编码 0 → 从 plan 字段读取）。

**具体工作项**：

1. **在 `TilingPlan` 新增两个字段**：
   ```python
   pool_addr_start: int = 0      # base_addr_pooling 的起始值（per-layer 常量）
   main_addr_start: int = 0      # base_addrs_res 的起始值（= layer_output_bas_addr）
   ```

2. **在 `tiling.py` 的 SD-UNet overrides 中填写具体值**（基于 golden 直接读取）：
   ```python
   # Layer idx=2 (c1_2, pool)
   {"pool_addr_start": 1152, "main_addr_start": 0, ...},     # 左 tile
   {"pool_addr_start": 1154, "main_addr_start": 576, ...},   # 右 tile
   ```

3. **在 `emitter._emit_w_macro_tile` 中使用新字段**：
   ```python
   # DS 循环内
   pool_addr_cur = plan.pool_addr_start + pool_iter_offset
   isa.DataStorer.dispatch(
       ...
       base_addr_pooling=pool_addr_cur,
       base_addrs_res=st.storer_bas_addr,
       ...
   )
   ```

4. **在 `addr_alloc.py` 中临时用正确的手工值覆盖**（可选，或直接让 emitter 从 plan 读）。

**阶段二（通用化）**：实施方案 C

在成功上板验证后，将硬编码值替换为 `compute_static_layout()` pass 自动推导，基于 `skip_sources` 信息和 TilingPlan 的 `load_total_num × storer_step × num_tiles` 公式。同时修复 `_output_size_words()` 使 Linear Scan 结果可靠。

---

## 七、预计工作量评估

| 任务 | 工作量 | 依赖 |
|------|--------|------|
| A1: TilingPlan 新增 pool_addr_start / main_addr_start 字段 | 0.5 天 | — |
| A2: 填写 SD-UNet 19 层的 override 值（4 个 pool 层 + decoder skip 读取地址） | 0.5 天 | A1 |
| A3: emitter.py 使用 plan.pool_addr_start（含 per-DS 递增逻辑分支） | 0.5 天 | A1 |
| A4: 测试：对比 golden，验证 base_addr_pooling 字段 | 0.5 天 | A3 |
| B1: 修复 `_output_size_words()` 使用 TilingPlan | 1 天 | 需先理解 bas_hint 含义 |
| B2: addr_alloc 新增 pool_addr_map 追踪 | 1 天 | B1 |
| B3: pipeline.py 传递 plans 给 addr_alloc | 0.5 天 | B2 |
| B4: emitter 接收 pool_addr_map | 0.5 天 | B3 |
| B5: 完整回归测试 | 1 天 | B4 |

**阶段一总计：约 2 天（阶段 A）**  
**通用化总计：约 5 天（A+B 完整实施）**

---

## 八、附录：Buffer A 静态布局总结

```
Buffer A Word Address Layout (h_in=144, w_in=256, 输入 1×1×144×256)
================================================================================
[    0,  1152)  c1_for_cat       (144×4×2 = 1152 words)  Layer 2 主输出
[ 1152,  1440)  c1_pool_out      (72×4    = 288  words)  Layer 2 pool 输出
[ 1152,  1728)  c3_for_cat       (72×8    = 576  words)  Layer 4 主输出  ← 覆盖 c1_pool_out 后半段？
                                                           [实际无重叠：c1_pool=[1152,1440), c3=[1152,1728)]
                                                           c1_pool 在 c3 写入前已无效 (valid=False)
[ 1728,  1872)  c3_pool_out      (36×4    = 144  words)  Layer 4 pool 输出
[ 1728,  2016)  c5_for_cat       (36×8    = 288  words)  Layer 6 主输出
[ 2016,  2088)  c5_pool_out      (18×4    = 72   words)  Layer 6 pool 输出（会被 c7_pool 覆盖）
[ 2016,  2160)  c7_pool_out      (9×8×2   = 144  words)  Layer 8 pool 输出（L1=0/1 各 72 words）
[ 2160,  2448)  c7_for_cat       (18×8×2  = 288  words)  Layer 8 主输出（L1=0/1 各 144 words）
================================================================================
总预留：2448 words（约 156 KB，以 64 位每字节换算）

注：c5_pool 与 c7_pool 重叠是有意为之——golden 将 c5_pool 的 valid 标记为 False 
    后再写 c7_pool，硬件不会同时读取两者。
```

---

## 九、关键技术发现总结

1. **`_output_size_words()` 误差 32×**：现有公式将 `cout` 乘入字数计算，但硬件 word 单位仅与像素宽度有关（64 px/word），与 cout 无关。正确公式：`h_out × ceil(w_out/64) × num_macro_tiles`。

2. **`base_addr_pooling` 系统性缺失**：emitter.py 在 4 处（L323/532/604/710）将该字段置 0，`addr_alloc.py` 完全未追踪 pool 输出地址。需新增独立的 `pool_addr_map`。

3. **Buffer A 是持久化 skip 仓库**：Golden 将 encoder 各层的 skip 张量累积写入 buffer A，各张量区域在各自最后一次被解码层读取前保持有效。Buffer B 仅用于"当前层"临时输出，不持久化。

4. **Linear Scan 架构方向正确**：活跃区间分析（`skip_sources` 驱动 `last_use` 延长）逻辑正确，只需修复大小公式即可使线性扫描产生正确的 `addr_map`。

5. **Pool 增量模式依赖 pool_output_mode**：
   - mode=0（Layer 2）：每 DS 步 +4
   - mode=1（Layer 4/6）：每 2 DS 步 +4
   - mode=2（Layer 8）：每 2 DS 步 +8
   
   这与 `plan.pool_output_mode` 字段一一对应，emitter 可从中派生正确的递增逻辑。
