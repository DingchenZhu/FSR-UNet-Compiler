# TVM 编译器前端设计 — 实现路线图

## Phase 1 范围（当前批次）

**In scope:**
- ONNX 和 PyTorch 双入口前端 → Relay IR
- Relay IR 无优化结构降级（InferType 除外）
- LayerDesc 中间表示提取
- TilingPlan 生成（覆盖 5 种标准空间尺寸）
- DeformableConv2d 一等公民路径（OffsetLoader + bilinear WeightLoader）
- 伪指令发射 + 依赖分析 + 虚拟寄存器分配
- 各阶段 dump 到 output/

**Not in scope (Phase 2):**
- 真实 TVM 后端编译（relay.build）
- 算子融合优化 Pass
- 量化感知训练后处理
- 自动 Tiling 搜索
- Word 文档生成

---

## 模块分工

```
tvm-design/
├── frontend/
│   └── frontend.py          # ONNX/PyTorch → Relay IRModule + params
├── ir/
│   └── layer_desc.py        # Relay IR → LayerDesc 列表
├── tiling/
│   └── tiling.py            # LayerDesc → TilingPlan
├── backend/
│   ├── isa.py               # 7类ISA指令包装器（golden格式兼容）
│   ├── emitter.py           # TilingPlan → 原始指令序列
│   └── post_pass.py         # 依赖分析 + 虚拟寄存器分配
├── pipeline.py              # 顶层编排，接收模型路径，输出到 output/
└── output/
    ├── relay_ir.txt         # Relay IR 文本 dump
    ├── layer_descs.json     # LayerDesc 列表
    ├── tiling_plan.json     # TilingPlan 列表
    └── pseudo_instructions.txt  # 最终伪指令（golden格式，一行一个dict）
```

---

## 数据结构

### LayerDesc
```python
@dataclass
class LayerDesc:
    op: str           # conv2d / deformable_conv2d / pool2d / relu / ...
    idx: int
    h_in, w_in: int
    cin, cout: int
    k_h, k_w: int = 1
    stride_h, stride_w: int = 1
    pad_top, pad_left, pad_bottom, pad_right: int = 0
    groups: int = 1
    deformable: bool = False
    deformable_groups: int = 1
    dilation_h, dilation_w: int = 1
    needs_pixel_shuffle: bool = False
    upscale_factor: int = 1
    pool_type: Optional[str] = None
    pool_size: Tuple[int, int] = (1, 1)
```

### TilingPlan
```python
@dataclass
class TilingPlan:
    layer_idx: int
    h_out_per_step: int       # 每轮外循环推进的输出行数（标准conv=2，deformable=4）
    load_total_num: int       # H方向DataLoader块数
    padding_num: int          # 首尾边缘块数
    line_buffer_rows: int     # 每次DataLoader加载的行数（4=standard，6=deformable）
    line_buffer_reshape: int  # 0/1/2/3 对应4种line buffer重排模式
    w_macro_tiles: List[...]  # [(w_start, w_size, bas_addr_hint), ...]
    w_micro_tile: int         # 32/64/128
    cin_group: int            # 输入通道分组（2/4/8ic）
    cout_group: int           # 输出通道分组（8/16/32oc）
    weight_parall_mode: int   # 0/1 MAC阵列上下半选择
    weight_transnum_base: int # WeightLoader transnum基数
    read_mode: int            # DataLoader read_mode
    use_bilinear_weights: int # 0/1 bilinear MAC模式
    ky_outer: int             # deformable: ky循环次数（3x3 kernel → 3）
    ic_inner: int             # deformable: 每ky的ic组数
```

### ISA 指令字段（golden格式）
所有指令均有 `code_num: [N]`（单元素列表）+ `op_code` + post_pass后附加的
`dependency`, `dest`, `src1`, `src2`, `src3`, `src4`。

| 指令类型 | 关键字段 |
|---------|---------|
| OffchipDataLoader | transnum, load_model, src_buffer_idx, bas_addr, is_compression=0 |
| DataLoader | layer_idx, line_buffer_reshape, is_padding_row, read_mode, transnum, line_buffer_idx, src_buffer_idx, bas_addr, offchip_read_mode=0, is_compression=0 |
| WeightLoader | acc_reg_comp_idx, kernal_size, line_buffer_row_shift, line_buffer_idx, is_padding_col, weight_parall_mode, is_new, transnum, bas_addr, is_bilinear_bicubic, offset_reg_idx, is_skip=2 |
| OffsetLoader | offset_reg_idx, bas_addr |
| QuantLoader | quant_reg_load_idx, quant_mode, layer_idx, transnum, bas_addr |
| DataStorer | quant_config_idx, pixelshuffle_out_mode, is_pixelshuffle, pooling_out_mode, pooling_out_new, is_pooling, reg_out_idx, acc_mode, transfer_num, store_mode, stride, base_addr_pooling, base_addrs_res, is_bicubic_add, is_first_or_last_row, is_mask, is_new, dest_buffer_idx, is_offset=0 |
| OffchipDataStorer | src_buffer, transnum, base_addr, is_compression=0 |

---

## 实现顺序

1. **backend/isa.py** — ISA包装器（最底层，无依赖）
2. **ir/layer_desc.py** — LayerDesc + Relay walker
3. **tiling/tiling.py** — TilingPlan生成规则
4. **backend/emitter.py** — 指令发射（依赖isa + layer_desc + tiling）
5. **backend/post_pass.py** — 依赖分析 + 寄存器分配（依赖emitter输出）
6. **frontend/frontend.py** — ONNX/PyTorch导入（最上层）
7. **pipeline.py** — 串联所有阶段，dump输出

---

## DeformableConv2d 完整路径

```
PyTorch nn.DeformableConv2d
    ↓ torch.jit.trace
    ↓ relay.frontend.from_pytorch
Relay: nn.deformable_conv2d(data, offset, weight, ...)
    ↓ extract_layer_descs → LayerDesc(op="deformable_conv2d", deformable=True)
    ↓ choose_tiling → TilingPlan(line_buffer_rows=6, ky_outer=3, ic_inner=2,
                                  use_bilinear_weights=1)
    ↓ _emit_deformable_conv:
        for cal_idx in range(cal_total):
            for ky in range(ky_outer=3):
                OffsetLoader(offset_reg_idx=...)
                for ic_g in range(ic_inner=2):
                    DataLoader(transnum=6, is_bilinear_bicubic=1)
                    WeightLoader(is_bilinear_bicubic=1, offset_reg_idx=...)
            DataStorer(is_pooling=1, pooling_out_mode=3, acc_mode=4)
    ↓ post_pass.finalize_instructions
7类ISA指令流（golden格式）
```

---

## Golden 对齐策略

1. **字段对齐** (`post_pass.align_instruction_fields`): 补全所有默认字段（is_skip=2等）
2. **依赖分析** (`add_instruction_dependencies`): 反向扫描，规则与sd_sr_codegen一致
3. **寄存器分配** (`assign_dependency_registers`): 15个虚拟寄存器池，LIFO释放

**验证方法:**
```bash
diff <(python3 -c "
import ast
for line in open('output/pseudo_instructions.txt'):
    line = line.strip()
    if line: print(line)
") <(head -n N golden/pseudo_code_load_next_first.txt)
```

---

## 已知风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| DeformableConv2d ONNX导出丢失offset分支 | 无法识别deformable | 优先用PyTorch路径 |
| 地址增量与golden不一致 | diff失败 | 逐层对齐sd_sr_codegen地址逻辑 |
| 寄存器分配顺序差异 | src字段不同 | 严格复现LIFO释放逻辑 |
| Relay InferType失败 | 流水中断 | 检查PyTorch trace shape是否正确 |
