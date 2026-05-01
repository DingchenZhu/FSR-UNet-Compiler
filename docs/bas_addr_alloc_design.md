# Feature Buffer Address Allocation — Polyhedral + ILP 设计文档

> 状态：设计 + 实现完成  
> 作者：编译器工程  
> 日期：2026-04-25

---

## 1. 问题背景

### 1.1 当前缺口

编译器已能对 FSRCNN（顺序拓扑）生成与 golden 完全对齐的指令流（0 diff）。
对 UNet 这类含 **skip connection** 的模型，DataLoader 指令中的 `bas_addr` 字段需要知道：
目标 tensor 在片上 feature buffer（a 或 b）中的**起始 word 地址**。

FSRCNN 之所以不需要分配器，是因为每层输出紧接前层写入位置（`bas_addr=0` 起步，tile 内逐行步进）。
UNet 的 skip tensor 需要在编码器写入后持续存活至解码器，与同一 buffer 内的其他 tensor **共存**，地址不能重叠。

### 1.2 涉及的字段

| 指令 | `bas_addr` 含义 |
|------|----------------|
| `DataLoader` | 片上 feature buffer（a/b）内读起点（word 地址） |
| `DataStorer` | `base_addrs_res`：同上，写起点 |
| `WeightLoader` | 权重 bank 内起点（由 emitter 独立管理，不在本文范围） |
| `QuantLoader` | 量化参数 bank 内起点（由 emitter 独立管理） |

**本文只处理 DataLoader/DataStorer 的 per-layer feature buffer bas_addr。**

---

## 2. 形式化模型

### 2.1 输入

每个参与分配的 tensor $t_i$ 描述为四元组：

$$t_i = (\text{buffer}_i,\ s_i,\ d_i,\ u_i)$$

- $\text{buffer}_i \in \{a, b\}$：由硬件 ping-pong 规则预先确定，分配器无权修改
- $s_i$：大小（words）
- $d_i$：定义层（def_layer，该层的 DataStorer 写入此 tensor）
- $u_i$：最后使用层（last_use，该层的 DataLoader 读完此 tensor）

两个 tensor 的**活跃区间重叠**定义为：$d_i \le u_j$ 且 $d_j \le u_i$（区间有任意交集）。

### 2.2 决策变量

$$x_i \in \mathbb{Z}_{\ge 0},\quad \forall i$$

$x_i$ 为 tensor $t_i$ 在其 buffer 内的起始 word 地址。

### 2.3 约束

**非重叠约束**（同一 buffer 且活跃区间重叠的任意两个 tensor）：

$$x_i + s_i \le x_j \quad \text{OR} \quad x_j + s_j \le x_i$$

引入二元变量 $y_{ij} \in \{0, 1\}$ 线性化（big-M 方法，$M = \sum_k s_k$）：

$$x_i - x_j + M \cdot y_{ij} \le M - s_i \tag{C1}$$
$$x_j - x_i - M \cdot y_{ij} \le -s_j \tag{C2}$$

**峰值约束**（引入辅助变量 $z$）：

$$x_i + s_i \le z,\quad \forall i \in \text{buffer } b \tag{C3}$$

### 2.4 目标

$$\min_{x, y, z} \quad z_a + z_b$$

其中 $z_a$/$z_b$ 分别是 buffer a 和 b 的峰值。

---

## 3. 多面体视角

将每个 tensor $t_i$ 看作时间-地址二维平面上的矩形：

- **x 轴（时间）**：$[d_i, u_i]$
- **y 轴（地址）**：$[x_i, x_i + s_i)$

非重叠约束等价于：**同一 buffer 中任意两个矩形不得在 y 轴上重叠**（时间轴有交叠时）。

这是一维区间调度的特殊情形，可用多面体工具（ISL / PPCG）表达为：
$$\forall (i, j),\ \text{buffer}_i = \text{buffer}_j,\ [d_i, u_i] \cap [d_j, u_j] \ne \emptyset \implies [x_i, x_i+s_i) \cap [x_j, x_j+s_j) = \emptyset$$

多面体分析的贡献在于能对复杂嵌套 live-range（如 UNet 的 4 层 skip）自动生成上述约束对集合，无需手写。

---

## 4. 等价性：ILP = Linear Scan（对 UNet 结构）

经 `ir/mem_alloc.py` 实测，对 USR-Net 32 层结构：

| 算法 | Buffer A | Buffer B | 总峰值 | vs 理论下界 |
|------|----------|----------|--------|-------------|
| Linear Scan | 16384w | 8192w | 24576w | **+0 (0%)** |
| TVM Workspace | 16384w | 8192w | 24576w | **+0 (0%)** |
| MLIR Bufferization | 16384w | 8192w | 24576w | **+0 (0%)** |
| ILP（理论最优） | 16384w | 8192w | 24576w | **+0 (0%)** |

**原因**：UNet 的 skip tensor 是**嵌套**而非**并列**的，同时存活，无复用机会。
数学上，当活跃区间不存在"并列不重叠"对时，贪心线性扫描即可达到全局最优。

**实际策略**：
- **默认路径**：Linear Scan（微秒级，对 UNet 等价最优）
- **严格路径**（`--alloc-solver ilp`）：scipy.optimize.milp（对任意拓扑精确最优，含未来的并列 skip 模型）

---

## 5. 实现架构

```
ir/addr_alloc.py
├── Tensor           — 输入描述（同 mem_alloc.py，含 name/size/def_layer/last_use/buffer）
├── AddressMap       — 输出 {layer_idx: bas_addr}，区分 a/b
├── LinearScanAllocator.allocate(tensors) → AddressMap
└── ILPAllocator.allocate(tensors)        → AddressMap  （需 scipy >= 1.7）
```

**与 emitter 的接口**：

```python
# pipeline.py
addr_map = allocate_addresses(layer_descs, tiling_plans)

# emitter.py — _start_layer() 内
bas_addr = addr_map.get(layer.idx, 0)
st.dataloader_bas_addr = bas_addr
st.storer_bas_addr     = addr_map.get_storer(layer.idx)
```

---

## 6. 技术风险

| 风险 | 等级 | 说明 | 缓解策略 |
|------|------|------|---------|
| **ILP 求解超时** | 低 | UNet 32 tensors → ~496 对约束，scipy milp 通常 < 1s；千层模型理论 NP-hard | 设置 timelimit=5s，超时自动降级到 Linear Scan |
| **big-M 值过大导致数值精度问题** | 低-中 | M = 总大小之和（~50K words），double 精度足够；但 M 越大 LP 松弛越弱 | 改用区间收紧约束（Tightened big-M）或 CP-SAT（若安装 ortools） |
| **活跃区间计算错误（skip 漏标）** | 中 | `last_use` 必须追踪到 decoder cat 层，否则地址提前被覆盖 | 用 `ir/layer_desc.py` 的 use-def 信息 + DFS DAG 遍历计算正确 last_use |
| **emitter 与分配器接口耦合** | 低 | 仅需在 `_start_layer` 注入 `bas_addr`，改动点集中 | 保留 `bas_hint` 参数通道，AddressMap 为可选 |
| **UNet golden bas_addr 参考值缺失** | 高 | 当前没有 UNet 全字段 golden（仅有结构对齐验证），ILP 输出无法直接 diff | 先运行分配器，与手工推算的 skip 地址对比；后续用 golden 文件验证 |

---

## 7. 技术收益

| 收益 | 说明 |
|------|------|
| **自动正确性** | skip tensor 地址由约束求解，消除手工计算出错风险 |
| **任意模型泛化** | 对含 residual、multi-branch、dilated skip 的模型同样适用，无需修改 |
| **内存最优保证** | ILP 路径提供精确最优（对 UNet 等模型等同于理论下界） |
| **UNet golden 验证解锁** | bas_addr 确定后，可对 UNet 指令流做全字段 diff |
| **编译器完整性** | 消除最后一个需外部手工配置的参数，编译器完全自包含 |
| **对未来模型的内存节省** | 若未来出现并列（非嵌套）skip，ILP 可比贪心算法节省数 KB 片上内存 |

---

## 8. 依赖与环境

| 依赖 | 版本 | 用途 | 可用性 |
|------|------|------|--------|
| `scipy.optimize.milp` | scipy >= 1.7 | ILP 求解 | ✅ scipy 1.17.0 已安装 |
| `ortools.sat.python.cp_model` | ortools >= 9 | CP-SAT 替代（更鲁棒） | ❌ 未安装，可选 |
| `numpy` | — | 矩阵构造 | ✅ |

Linear Scan 路径无额外依赖。

---

## 9. 实现路线图

| 步骤 | 内容 | 预计复杂度 |
|------|------|-----------|
| 1 | `ir/addr_alloc.py`：Linear Scan + ILP 分配器 | 中 |
| 2 | `ir/layer_desc.py`：添加 `last_use` 字段推导（DFS） | 低 |
| 3 | `pipeline.py`：调用分配器，生成 `AddressMap` | 低 |
| 4 | `backend/emitter.py`：`_start_layer` 接收 `AddressMap` | 低 |
| 5 | 运行 UNet 编译，与手工 skip 地址对比验证 | 中 |
| 6 | FSRCNN 回归测试（必须保持 PERFECT） | 低 |
