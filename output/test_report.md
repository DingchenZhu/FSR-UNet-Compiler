# 编译器测试报告

**日期：** 2026-04-22  
**测试环境：** TVM 0.8.dev0 / torch 2.10.0+cu128 / onnx 1.20.1

---

## 测试结果汇总

| 测试项 | 结果 | 说明 |
|--------|------|------|
| 环境检查 | ✅ PASS | TVM/torch/onnx 版本均可用 |
| 模块导入 | ✅ PASS | 所有模块无错误导入 |
| ISA 单元测试 | ✅ PASS | 指令字段格式与 golden 一致 |
| FSRCNN ONNX 导入 | ✅ PASS | fsrcnn_simplified.onnx 可正常加载 |
| USR_Net 完整流水线 | ✅ PASS | 98层/7298条指令生成成功 |
| Golden 对比 | ⚠️ PARTIAL | 结构匹配，但调度模式不同 |

---

## Golden 溯源分析（关键发现）

Golden 文件由 `sd_sr_codegen.py` 的 `sd_inst(is_first=True, load_next=True)` 生成：

- `pseudo_code_load_next_first.txt` = 第一帧（is_first=True, load_next=True）
- `pseudo_code_load_next_mid.txt` = 中间帧（is_first=False, load_next=True）
- `pseudo_code_load_next_last.txt` = 最后帧（is_first=False, load_next=False）

**全部3个文件均为 UNet（stable diffusion）pipeline 的指令流，不是单独的 FSRCNN。**

sd_inst 生成 17160 条原始指令（code_num [0]-[17159]），golden 文件包含 [5]-[17159]，截去了最前面的 5 条 OffchipDataLoader preamble。

**关键验证：**
- sd_inst op_code 序列与 golden 完全吻合（20/20 连续匹配 ✓）
- 我们的流水线产生 7298 条 vs golden 17156 条

---

## 两者差距原因

### 1. load_next 调度模式（最大差距来源）

sd_codegen 实现了"加载下一层"预取调度：
- 在处理 layer N 的 tile 时，提前发射 layer N+1 的 QuantLoader
- 这使得 QuantLoader 与 DataLoader/WeightLoader 并发执行
- 我们的 vis_compiler 使用顺序调度（QuantLoader 在本层所有 tile 之前）

```
Golden 调度:            我们的调度:
QuantLoader(layer 1)    QuantLoader(layer 0)
DataLoader(layer 0)     DataLoader(layer 0)
WeightLoader(layer 0)   WeightLoader(layer 0)
DataStorer(layer 0)     DataStorer(layer 0)
QuantLoader(layer 2)    DataLoader(layer 0)
DataLoader(layer 0)     ...（所有 layer 0 tile）
...                     QuantLoader(layer 1)
                        DataLoader(layer 1)
```

### 2. line_buffer_idx 切换 bug

sd_codegen 使用独立的 `dataloadermanager` 和 `weightloadermanager`，各自独立 toggle：
- 结果：DataLoader 和 WeightLoader 在每次迭代中使用**相同的** line_buffer_idx
- 我们的实现：在 DataLoader 之后、WeightLoader 之前 toggle，导致两者 idx 始终相反

```python
# sd_codegen（正确）:
DataLoader(line_buffer_idx=0)  → toggle → DL.idx=1
WeightLoader(line_buffer_idx=0)  → toggle → WL.idx=1  # 用的是 WL 自己的初始值

# 我们的实现（错误）:
DataLoader(line_buffer_idx=0)  → toggle st.lb=1
WeightLoader(line_buffer_idx=1)  → toggle st.lb=0  # 用了切换后的值
```

### 3. 缺少 OffchipDataLoader preamble

Golden 前5条是 DDR→片上缓冲的预加载指令，我们的 emitter 没有发射这5条：
- quant DDR load (load_model=0, src_buffer_idx=2)
- quant DDR load (load_model=1, src_buffer_idx=2)
- weight DDR load (load_model=0, src_buffer_idx=1)
- weight DDR load (load_model=1, src_buffer_idx=1)
- weight DDR load (load_model=2, src_buffer_idx=1)

### 4. 虚拟寄存器编号不同（次要问题）

Golden 的 post_pass 是对全部 17160 条指令运行的，寄存器分配从 [0] 开始。
我们的 post_pass 只处理我们生成的指令，编号起点不同。
需要包含所有前置指令才能重现相同的寄存器编号。

### 5. is_new 字段版本差异

sd_codegen 当前代码：`is_new=0`（覆盖模式）  
Golden 显示：`is_new=1`（累加模式）  
→ Golden 是由较早版本的 sd_codegen 生成的，`is_new` 语义存在版本差异

---

## 修复优先级

| 优先级 | 修复项 | 预估影响 |
|--------|--------|---------|
| P0 | 修复 line_buffer_idx 切换 | 修复所有 DataLoader/WeightLoader 字段 |
| P0 | 添加 OffchipDataLoader preamble (5条) | 修复 code_num 偏移 |
| P1 | 实现 load_next QuantLoader 预取调度 | 修复调度顺序，指令数增至约 17160 |
| P1 | 修复 is_new 字段 (0→1 或 按版本对齐) | 修复 WeightLoader 字段 |
| P2 | 修复 layer_idx 编号（从1开始） | 修复 QuantLoader/DataLoader layer_idx |

---

## 当前有效输出

`output/pseudo_instructions.txt` — 7298 条指令，结构正确但调度模式为顺序（非 load_next）。
可作为调试中间参考。

---

## 下一步行动

1. **修复 emitter.py**：`line_buffer_idx` 不在 DataLoader 和 WeightLoader 之间切换，改为只在 WeightLoader 之后切换一次
2. **添加 preamble**：在 pipeline.py 中添加 5 条 OffchipDataLoader 初始化指令
3. **实现 load_next**：在 emitter 中为每层第一个 tile 添加下一层 QuantLoader 预取
4. **对齐 is_new**：WeightLoader 的 is_new 字段改为 1（与 golden 一致）
