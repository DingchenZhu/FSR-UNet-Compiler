# 编译器架构建议文档

> 由 tvm-compiler-expert 生成，供 compiler-coder 实现参考

---

## 1. 整体流水线设计

推荐五阶段线性流水：

```
Input Model
    ↓
[Frontend]  ONNX/PyTorch → Relay IRModule + params
    ↓
[IR Stage]  Relay → LayerDesc[]  (InferType only, no fusion/fold)
    ↓
[Tiling]    LayerDesc → TilingPlan[]  (template registry)
    ↓
[Emitter]   TilingPlan → raw instruction dicts  (7 ISA types)
    ↓
[PostPass]  field alignment + dependency edges + virtual register alloc
    ↓
pseudo_instructions.txt  (golden-compatible format)
```

各阶段接口契约：
- Frontend: `(model_path, input_shapes) → (IRModule, params)`
- IR Stage: `(IRModule) → List[LayerDesc]`
- Tiling: `(List[LayerDesc]) → List[TilingPlan]`
- Emitter: `(List[LayerDesc], List[TilingPlan]) → List[Dict]`
- PostPass: `(List[Dict]) → List[Dict]` (in-place mutation)

---

## 2. 前端设计

### ONNX 路径
```python
mod, params = relay.frontend.from_onnx(
    onnx.load(path),
    shape={name: shape},
    dtype={"input": "float32"},
    freeze_params=True
)
```

Phase 1 允许的 Relay Pass（仅这些，其他均不运行）：
- `relay.transform.InferType()` — 必须，形状推断
- `relay.transform.FoldConstant()` — 可选，仅在 TVM LLVM 可用时

**不允许：** FuseOps, SimplifyExpr, ToANormalForm（会改变图结构影响golden parity）

### PyTorch 路径
```python
model.eval()
traced = torch.jit.trace(model, example_inputs)
mod, params = relay.frontend.from_pytorch(
    traced,
    input_infos=[("input", input_shape)],
    default_dtype="float32",
    use_parser_friendly_name=True,
)
```

**注意：** DeformableConv2d 在 PyTorch trace 时需要 `torchvision.ops.deform_conv2d`，
TVM from_pytorch 会将其识别为 `nn.deformable_conv2d`（需确认 TVM 版本支持）。

---

## 3. DeformableConv2d 处理

### 为何不能用 TVM 通用降级
TVM 通用降级会将 `nn.deformable_conv2d` 展开为基础张量运算（gather、scatter等），
无法映射到硬件 OffsetLoader 指令。硬件直接支持 offset-based addressing，
必须保留算子语义。

### 推荐处理方式
在 `extract_layer_descs` 中识别 `nn.deformable_conv2d` op，设置 `deformable=True`，
通过 `choose_tiling` 路由到 deformable 路径，在 `_emit_deformable_conv` 中
发射 `OffsetLoader + DataLoader(6row) + WeightLoader(bilinear)` 序列。

**关键：** 不要在 Relay Pass 阶段对 deformable_conv2d 做任何变换。

---

## 4. Tiling 策略架构

### 模板注册表（关键修正）

当前 `choose_tiling()` 中有以下已知问题，必须修正才能达到 golden parity：

1. **`line_buffer_reshape` 应为 1**（对于 128×72 及更小的层），不能总是 0
2. **`weight_transnum_base` 需要按实际 cin 计算**，layer 3+ 需要 transnum=12
3. **`TilingPlan` 缺少以下字段**，必须添加：
   - `acc_mode` — DataStorer.acc_mode（0=standard，4=deformable pooling）
   - `store_mode` — DataStorer.store_mode（0=standard，3=deformable）
   - `quant_mode` — QuantLoader.quant_mode（逐层不同）
   - `quant_transnum` — QuantLoader.transnum（逐层不同）
   - `weight_total_transnum` — 每层权重总大小（用于地址推进）

### 推荐注册表结构
```python
_TEMPLATE_TABLE = {
    # (h_in, w_in, cin, cout, k_h, deformable) → template_params_dict
    (144, 256, 1, 32, 3, False): {...},
    (144, 128, 32, 32, 3, False): {...},
    ...
}
```

精确匹配优先，fallback 到启发式规则。

---

## 5. ISA 发射架构

### 地址计算原则（去除魔数）

取代硬编码地址增量（如 288、144*4），推荐预先计算 BufferLayout：

```python
@dataclass
class BufferLayout:
    input_base: int
    input_right_half_offset: int   # w_in * cin for 256-wide split
    output_base: int
    output_right_half_offset: int  # h_in * 4 (from sd_codegen)
    weight_base: int
    quant_base: int
```

在 `emit_layer` 调用前由 pipeline 计算，传入 emitter。

### line_buffer_idx 切换顺序（严禁改动）
```python
DataLoader.dispatch(..., line_buffer_idx=st.line_buffer_idx)
st.line_buffer_idx = 1 - st.line_buffer_idx   # ← DataLoader 后立即切换
WeightLoader.dispatch(..., line_buffer_idx=st.line_buffer_idx)  # 用切换后的值
st.line_buffer_idx = 1 - st.line_buffer_idx   # ← WeightLoader 后再切回
```
此顺序与 sd_codegen/sd_sr_codegen 完全一致，**任何重构均不得改变**。

---

## 6. Golden Parity 策略

### 必须 bit-exact 的字段
- `dependency` 列表（指令索引）
- `dest`, `src1`, `src2`, `src3`, `src4`（虚拟寄存器）
- 所有默认字段（`is_skip=2`, `is_offset=0`, `is_compression=0`等）

### 可以有实现定义差异的字段
- 无——所有字段均需匹配

### 验证流程
```python
import ast
golden = [ast.literal_eval(l) for l in open("golden/pseudo_code_load_next_first.txt") if l.strip()]
output = [ast.literal_eval(l) for l in open("output/pseudo_instructions.txt") if l.strip()]
for i, (g, o) in enumerate(zip(golden, output)):
    if g != o:
        print(f"Mismatch at instruction {i}: {g} != {o}")
```

---

## 7. Post-Pass 关键细节（严禁重构）

### src4 quirk（必须保留）
```python
code_dict["src4"] = src_code[2] if len(src_code) > 3 else 0
# 注意：这里是 src_code[2] 不是 src_code[3]
# 这与 sd_sr_codegen.py 的原始实现一致，golden依赖此行为
```

### 依赖规则（7条，逐条照搬 sd_sr_codegen）
不要泛化！每条规则对应特定 op_code，且有精确的 backward scan 停止条件。
任何泛化都会导致某些指令依赖边缺失或多余，破坏寄存器分配。

---

## 8. 测试策略

### 单元测试
- `test_isa.py`: 每种指令类型的 dispatch + 字段验证
- `test_layer_desc.py`: 对已知 ONNX/PyTorch 模型验证 LayerDesc 字段
- `test_tiling.py`: 对 5 种标准尺寸验证 TilingPlan
- `test_post_pass.py`: 对小型手写指令序列验证依赖边和寄存器分配

### 端到端测试
- 对 FSRCNN（PyTorch）运行完整流水，diff 输出与 `golden/pseudo_code_load_next_*.txt`
- 对 USR_Net（ONNX）运行完整流水，diff 输出与对应 golden

### 中间 dump 调试
当 diff 失败时，比较各阶段 dump：
1. `relay_ir.txt` — 确认算子识别正确
2. `layer_descs.json` — 确认形状/padding/deformable标志
3. `tiling_plan.json` — 确认tiling参数与预期一致
4. 逐条比对指令，定位第一条不匹配的指令

---

## 总结优先级

| 优先级 | 修复项 |
|--------|--------|
| P0（阻塞golden parity）| TilingPlan 缺少 acc_mode/store_mode/quant_transnum 字段 |
| P0 | line_buffer_reshape 需按层设置，不能全为0 |
| P0 | weight_transnum_base 精确计算 |
| P1（影响正确性）| BufferLayout 预计算（去除地址魔数） |
| P1 | DeformableConv2d ONNX支持验证 |
| P2（工程质量）| _finalize_layer 状态推进集中管理 |
| P2 | 模板注册表精确匹配 |
