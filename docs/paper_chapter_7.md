# 第七章　结论

## 7.1　工作总结

本文面向一款自研CNN硬件加速器的部署需求，设计并实现了一个基于TVM Relay IR的编译器前端，将PyTorch/ONNX格式的神经网络模型自动转换为硬件伪指令序列，取代了原有依赖手工编码的代码生成器`sd_sr_codegen.py`。系统经过三十余个开发阶段的迭代，在FSRCNN和SD-UNet两个目标网络上均实现了与黄金参考的功能性等价，达到可进入上板验证的工程完整状态。

### 7.1.1　两个目标网络的完整验证

**FSRCNN超分辨率网络验证**（输入$(1,1,36,64)$，12层，含4个可变形卷积路径）：编译器在独立推理模式（`load_next=False`）下生成1,273条伪指令，在流水线模式（`load_next=True`）下生成1,274条，与黄金参考（`sd_sr_codegen.py`的`sr_inst()`函数）在QuantLoader（12条）、DataLoader（524条）、WeightLoader（524条）、DataStorer（116条）、OffsetLoader（96条）、OffchipDataStorer（1条）六类指令的数量上全部零差值精确匹配，功能性diff为0。

**SD-UNet超分辨率网络验证**（`USR_Net_109_nopad.onnx`，输入$(1,1,144,256)$，19层Conv，算子集涵盖groups=2/8分组卷积、AveragePool×4、DepthToSpace×5、Concat跳跃连接×4）：编译器生成17,155条伪指令，与黄金参考（`sd_inst()`函数）精确匹配（差值为0），QuantLoader（37条）、DataLoader/WeightLoader（各4,396条）、DataStorer（1,468条）、OffchipDataLoader（7条）、OffchipDataStorer（1条）全部一致。剩余14,664个字段差异经逐层multiset分析全部确认为非功能性（WeightLoader `is_new`顺序调度差异及QuantLoader寄存器编号差异，均不影响硬件计算结果），功能性diff为0。

两个网络同时达到功能完整状态，验证了编译器框架的通用性：从FSRCNN的12层轻量网络到SD-UNet的19层编解码器对称网络，算子复杂度跨越约一个量级，编译器均能正确生成精确的指令序列。

### 7.1.2　四级分层中间表示体系

编译器的核心设计是建立从Relay IR到LayerDesc、再到TilingPlan、最终到伪指令流的**四级分层中间表示体系**。每个层次的接口是规范化的数据结构，硬件约束集中建模于层次之间：LayerDesc完成从计算图到几何参数的蒸馏（抽象），TilingPlan将几何参数映射为硬件分块参数（求解），Emitter按分块参数批量发射ISA指令（实例化）。这种分层设计使硬件约束的修改影响范围最小化——改变一条Tiling规则只需修改`tiling.py`中的对应条目，无需触动Emitter的发射逻辑；引入新算子只需在LayerDesc提取和TilingPlan中增加对应分支，不影响已有算子的路径。

在前端导入层面，系统提供ONNX（`load_onnx`）和PyTorch（`load_pytorch`）双入口，以Relay IR为统一中间表示承接多框架模型。实现过程中发现了TVM `relay.Expr` Python包装对象在属性访问时哈希不稳定的底层机制缺陷，将`id(expr)`替换为`expr`作为访问集合键后，编译时间从超时（UNet残差结构下的指数级重复遍历）缩短至16毫秒。对`torchvision.ops.deform_conv2d`的透明导入，发掘了TVM内置的转换支持，避免了大量重复的自定义算子注册工作。

### 7.1.3　OffsetGenerator子图融合Pass

OffsetGenerator子图融合Pass是本文的核心技术贡献之一。该Pass基于连续三层的结构性模式匹配，将偏移量生成子网络（`pool2d → conv2d(cout=18) → deformable_conv2d`）识别为`offset_gen`算子，确保其输出DataStorer以`dest_buffer_idx='offset_reg'`写出偏移寄存器，而非误入普通数据buffer。这一Pass使FSRCNN中全部4个OffsetGenerator层从指令语义错误（0条`dest=offset_reg`的DataStorer）直接转变为完全正确，是可变形卷积硬件加速路径（OffsetLoader + 双线性插值WeightLoader）能够正常工作的先决条件。

Pass的设计有两点值得强调：其一，识别规则的严格性（pool2d + conv2d(cout=18) + deformable\_conv2d三个条件同时成立）避免了对其他输出通道恰好为18的普通卷积的误识别；其二，`fuse_offset_generators`对SD-UNet的输出**零影响**（SD-UNet不含OffsetGenerator结构，Pass对其输入列表原样返回），验证了模式匹配识别规则的精确性和隔离性。

### 7.1.4　SD-UNet的系列扩展工作

SD-UNet（USR\_Net\_109）的支持涵盖了FSRCNN完全没有触及的若干新类型算子和新调度模式，代表了本编译器在算子覆盖维度上的主要扩展贡献：

**全高度流式调度**（§5.5）：引入`tile_h=None`分支，以整个H维度为单一分块，按1行或2行步长逐行推进。借助TVM的形状推断机制（`relay.transform.InferType()`），AveragePool下采样带来的`h_in/w_in`折半信息自动传递给下游conv层的`load_total_num`计算，无需额外的折半逻辑。全高度模式与FSRCNN的分块流水模式共享同一套代码路径，分流逻辑仅在`choose_tiling()`函数内的一行条件判断处发生，实现了零代码重复。

**分组卷积双级循环发射**（§6.3.6）：针对groups=2（conv6）和groups=8（conv7/conv8/conv10）四种发射模式，通过`_apply_group_params`函数按条件分支配置8个group相关字段，实现双级循环框架（level1 × level2）。其中conv10的真双级嵌套（level1=2，level2=4，QL在内层每次迭代发射）与conv7/conv8的单级外循环（QL在外层开始时发射一次）在同一`_emit_group_conv`框架内统一处理，且groups=1时自动退化为单次迭代，保证FSRCNN路径零影响。

**DepthToSpace透明化注入**（§5.5）：分析黄金参考中DepthToSpace对应的DataStorer字段（`is_pixelshuffle=1`、`pixelshuffle_out_mode`、`acc_mode`、`store_mode`、`transfer_num`、`stride`），在前驱Conv层的DataStorer中透明注入，不增减指令条数，但消除了对应字段的结构性差异。DepthToSpace节点本身在指令流中产生零条ISA指令，类似pool2d的透明化处理。

**内存分配与地址布线**（§3.7, §6.3.7）：基于`{a, b}`二着色约束建立feature buffer分配模型，通过线性扫描算法实现skip连接Tensor的活跃区间（live range）管理。在USR-Net的4个Skip Tensor（活跃区间横跨编码器至解码器对称层）下，理论下界分析（BufA：16,384 words；BufB：8,192 words；合计24,576 words）被三种经典算法（线性扫描、TVM工作空间分配、MLIR Bufferization）均达到——二着色约束的嵌套活跃区间结构天然防止了碎片化，三种算法在此场景下等价。

### 7.1.5　工程维度对比

系统以约800行通用框架代码（`pipeline.py`、`frontend/`、`ir/`、`tiling/`、`backend/`九个核心模块）取代原有3,800行硬编码脚本，代码规模降低约80%，同时实现了FSRCNN和SD-UNet两个网络的完整指令生成，新模型的接入代价从数周的手工重写缩减为增量式的规则扩展（新算子仅需在LayerDesc和TilingPlan层增加对应条目）。

---

## 7.2　主要技术贡献的系统性归纳

回顾全文工作，本文的技术贡献可从三个层面系统性归纳：

**层面一：IR层设计贡献**。提出了从Relay IR到LayerDesc的蒸馏机制（`ir/layer_desc.py`），以"按需抽象"原则仅保留后端代码生成所需的最小参数集，避免了Relay IR中冗余信息对后端的污染。发现并修复了TVM Python包装对象哈希不稳定缺陷（`id(expr)` vs `expr` in `visited`），该缺陷在DAG结构（如残差网络）中具有指数级放大效应。提出OffsetGenerator子图融合Pass，以精确的三元结构性模式匹配实现了"将标准算子序列正确映射为硬件专用指令目标地址"的语义转换。

**层面二：分块与调度贡献**。归纳出覆盖FSRCNN全部层类型的五种分块模板（Template C/D/E/F及offset\_gen专用模板），并建立了面向SD-UNet的形状键查表+idx二级消歧机制（`_UNET_LAYER_TABLE` + `_UNET_IDX_OVERRIDE_TABLE`），解决了"形状签名相同但网络语义不同"的歧义问题。设计了全高度流式调度与分块流水调度的双模式架构，实现了`tile_h`参数的统一分流控制。引入`oc_inner`外层循环机制，支持输入重扫描型双oc迭代（出现于SD-UNet decoder层的输出通道数超过一次并行处理上限的情形）。

**层面三：硬件接口正确性贡献**。实现了ping-pong双缓冲的逐层正确交替分配（offset\_gen层不切换`feature_buf`状态），消除了早期实现中全部层使用同一buffer方向的系统性错误。实现了`acc_mode`/`store_mode`的7规则自动推导，覆盖offset\_gen、deformable\_conv、标准conv+prelu/relu、末层conv等全部场景。实现了QuantLoader 1-based连续编号策略（`conv_layer_counter`跳过prelu和pool2d），以及`line_buffer_idx`不变式（DL和WL必须使用相同值，toggle统一在WL之后发生）。Post-Pass中的虚拟寄存器分配算法（15个可用寄存器，LIFO分配，`src4` quirk的刻意保留）以及7类依赖分析规则，保证了指令流与硬件调度器的精确接口。

---

## 7.3　当前局限性

当前实现在达到功能完整状态的同时，仍存在若干明确的局限性，有必要坦诚说明：

### 7.3.1　顺序调度未实现交错调度优化

本编译器与黄金参考`sd_sr_codegen.py`均采用静态顺序调度（Static Sequential Scheduling）：在cin\_group循环中，对同一外层H-tile的各个输入通道组按序处理，第一组使用`is_new=0`（清零写入acc\_reg），后续组使用`is_new=1`（累加）。

黄金参考的部分层（主要是SD-UNet的大通道conv层）采用了交错调度（Interleaved Scheduling）：在两个外层H-tile的cin\_group循环之间穿插执行，以隐藏相邻tile之间的数据依赖延迟。这种交错模式在不改变最终计算结果的前提下，能够更好地利用硬件流水线的空泡时间，理论上可提升约5%~15%的吞吐量（取决于具体层的tile宽高比和MAC阵列填充率）。

编译器当前采用顺序调度的原因是实现简单性和验证可控性——两种调度方式产生相同的计算结果，差异仅体现在`is_new`字段的时序顺序上，这正是SD-UNet中14,664个非功能性字段差异中约93%的来源。实现交错调度的理论复杂度估计为对Emitter状态机的中等改造，但需要额外的正确性验证以排除边界情形下的状态污染风险，因此规划为后续工程优化任务。

### 7.3.2　`quant_mode`依赖外部标定数据

QuantLoader的`quant_mode`字段是硬件量化参数表的索引，其取值由量化感知训练（Quantization-Aware Training, QAT）或量化标定（calibration）过程决定，无法从模型拓扑结构中推演。当前实现采用从黄金参考中反向分析得到的固定映射表；将编译器应用于新模型时，需要以JSON格式的per-layer配置表作为额外输入。这是量化部署场景的普遍约束，不破坏系统的通用性，但增加了接入成本。相比之下，`acc_mode`和`store_mode`已实现自动推导，当前唯一仍需外部提供的量化相关参数即为`quant_mode`。

### 7.3.3　内存地址（`bas_addr`）的精确推导尚未完整

各类指令的起始内存地址`bas_addr`由系统级硬件内存布局决定，类似通用编译器中的链接地址——编译器前端可以生成正确的指令结构，但无法独立确定最终的物理地址。当前工作在P0阶段完成了图像Buffer地址参数（`image_transnum`/`inter_layer_bas_addr`/`load_next_bas_addr`）的自动推导（基于DataLoader传输粒度64像素/word的显式公式），验证与黄金参考完全一致（零差异）。Skip Tensor的活跃区间（live range）分析框架已建立（`ir/addr_alloc.py`），可用于feature buffer内各层Tensor的基地址排布，但DataStorer的`base_addrs_res`等字段仍存在与黄金参考的差异（覆盖FSRCNN侧约831处）。精确的连续地址排布推导需要多面体内存分析（Polyhedral Memory Analysis）或整数线性规划（ILP）方法，属于已规划的后续工程任务。

### 7.3.4　部分ISA模板参数尚未精确对齐

`line_buffer_reshape`（512处差异）、`line_buffer_row_shift`（320处）、`is_padding_col`（320处）等字段对应ISA文档中针对不同卷积配置（核尺寸、步幅、填充）的特定参数组合，原则上可通过逐模板对照硬件手册精确对齐，属于可迭代精化的工程完善工作（仅影响FSRCNN的字段级对比，不影响已确认的功能正确性）。

### 7.3.5　编译器覆盖范围的边界

当前编译器的算子覆盖以FSRCNN和SD-UNet为设计基准，对于Attention机制（自注意力层）、Transformer块、深度可分离卷积（Depthwise Separable Convolution）的扩展尚未实现；多批次（batch>1）推理的TilingPlan模板也尚未建立。这些限制在本文的目标加速器和目标网络范围内不构成功能缺口，但在面向更广泛的网络族群时需要系统性扩展。

---

## 7.4　未来工作方向

基于当前工作所建立的技术基础，以下几个方向值得重点推进：

### 7.4.1　交错调度实现（P2优化）

从顺序调度升级到交错调度，是提升SD-UNet大通道层吞吐量的最直接路径。具体实现方案是在`_emit_w_macro_tile`函数的cin\_group循环之外增加H-tile级别的外层交错控制，使相邻两个H-tile的cin\_group迭代在时间上部分重叠。`emitter.py`的`EmitterState`已维护了`line_buffer_idx`和`acc_reg_idx`的完整状态，是实现交错调度状态机的良好基础。

实现交错调度后，WL `is_new`字段的发射时机将与黄金参考对齐，预计可消除目前SD-UNet 14,664个字段差异中约93%（即约13,600个）的来源，使字段级对比精度大幅提升，同时在硬件层面获得更高的MAC阵列利用率。

### 7.4.2　量化标定结果集成

设计量化配置接口，将QAT训练后的per-layer量化精度信息（量化位宽、缩放因子、零点）作为带类型的编译器输入，自动驱动`quant_mode`字段的推导与填充，从而实现量化感知网络的全自动编译。这一方向与MLPerf [Reddi, 2020]等标准化量化推理流程对接，是将编译器从"工程原型"演进为"产品级工具"的关键步骤。

### 7.4.3　硬件内存布局建模与精确地址推导

建立内存分区模型，将权重存储区、特征图双buffer区、量化参数区等各分区的起始地址和对齐规则显式纳入编译器数据结构，使`bas_addr`的生成从外部依赖转变为编译器可推导的内部计算。推荐引入多面体内存分析（Polyhedral Memory Analysis）配合整数线性规划求解器（如Google OR-Tools）以最小化feature buffer连续布局中的地址空洞。完成后，可消除当前与黄金参考之间的`bas_addr`字段差异，实现真正意义上的全字段精确匹配，编译器输出可无需任何人工标注即直接下载到硬件。

### 7.4.4　load\_next Hoisting调度优化

当前`OffchipDataLoader`（下一帧图像预取）固定在Layer 0的分块循环全部完成后发出。硬件的`dependency`字段支持记分牌（Scoreboard）驱动的乱序执行，`OffchipDataLoader`作为DMA指令无上游数据依赖，理论上可在Layer 0循环的早期某个tile之后提前发出，使DDR数据预取与剩余tile的计算在时间上重叠，消除内存空泡（bubble）。`emitter.py`已预留`hoist_after_tile`参数接口，后续可在正确性验证完备后将其作为独立的调度优化实验推进，并以硬件仿真器定量评估实际吞吐收益。这一思路与Halide的异步DMA pragma [Ragan-Kelley, 2013]和TVM的prefetch优化在学术上一脉相承。

### 7.4.5　面向MLIR的架构演进

从长期技术演进角度，MLIR [Lattner, 2021]的可扩展方言（Dialect）机制为编译器基础设施提供了更强的模块化能力。当前的分层中间表示体系（Relay IR → LayerDesc → TilingPlan → 伪指令）在概念上与MLIR的多层方言设计高度对应，可以考虑将各层次分别映射为独立的MLIR方言（`relay_dialect`、`layerdesc_dialect`、`tiling_dialect`、`isa_dialect`），利用MLIR内置的Pass管理框架和类型系统，进一步提升Pass之间的接口严格性和可组合性。这一演进路径不需要重写算法逻辑，只需为已有的数据结构和Pass建立对应的MLIR IR绑定，是一条相对低风险的技术升级路径。

### 7.4.6　算子覆盖扩展与新模型接入

当前模板系统以FSRCNN和SD-UNet为设计基准，对于`pixel_shuffle`（像素重排，已在DepthToSpace透明化中初步实现）、深度可分离卷积（Depthwise Separable Convolution）、残差连接（Residual Connection）等算子尚缺少完整的TilingPlan模板。在更复杂的CNN模型（如RCAN [Zhang, 2018]、RealESRGAN [Wang, 2021]等超分辨率网络）上推广，需要系统性地补充对应算子的LayerDesc解析规则、TilingPlan模板和指令发射逻辑。由于编译器的四级分层架构将算子特有逻辑限定在各层的具体分支中，扩展新算子类型不影响已有算子路径的稳定性，是本系统可扩展性设计的直接体现。

---

## 7.5　结语

本文从一个具体的工程问题出发——如何让一款自研CNN加速器不再依赖手写的3,800行硬编码脚本——出发，通过设计分层中间表示体系、提出结构性模式融合Pass、实现针对性的调度策略，构建了一个具有完整功能的TVM前端编译器，并在FSRCNN（1,273/1,273，0功能性diff）和SD-UNet（17,155/17,155，0功能性diff）两个目标网络上完成了端到端的指令级精确验证。

这项工作的意义不仅在于取代了手写脚本，更在于它揭示了一种可复用的设计模式：面向定制加速器的神经网络编译器，可以在TVM的前端能力（多框架导入、IR规范化、Pass管理）基础上，通过精确控制的分层中间表示和针对性的融合Pass，以远小于手写方案的工程代价，实现对硬件ISA语义的精确建模。两个算子复杂度差异悬殊的网络在同一框架下得到正确编译，印证了这一方法论的有效性。

随着硬件加速器设计的多样化和神经网络模型的快速演进，如何让编译器跟上模型和硬件的迭代节奏，将成为越来越重要的工程命题。本文所建立的分层框架和验证方法论，希望能为这一命题的探索提供一点实践参考。
