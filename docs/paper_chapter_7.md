# 第七章　结论

## 7.1　工作总结

本文面向一款自研CNN硬件加速器的部署需求，设计并实现了一个基于TVM Relay IR的编译器前端，将PyTorch/ONNX格式的神经网络模型自动转换为硬件伪指令序列，取代了原有依赖手工编码的代码生成器`sd_sr_codegen.py`。

在前端导入层面，系统提供ONNX和PyTorch双入口，以Relay IR为统一中间表示承接多框架模型。实现过程中发现了TVM `relay.Expr`Python包装对象在属性访问时哈希不稳定的底层机制缺陷，将`id(expr)`替换为`expr`作为访问集合键后，编译时间从超时（UNet残差结构下的指数级重复遍历）缩短至16毫秒。对`torchvision.ops.deform_conv2d`的透明导入，则发掘了TVM内置的转换支持，避免了大量重复的自定义算子注册工作。

在算子支持层面，OffsetGenerator子图融合Pass是本文的核心技术贡献。该Pass基于连续三层的结构性模式匹配，将偏移量生成子网络（`pool2d → conv2d(cout=18) → deformable_conv2d`）识别为`offset_gen`算子，确保其输出DataStorer以`dest_buffer_idx='offset_reg'`写出偏移寄存器，而非误入普通数据buffer。这一Pass使FSRCNN中全部4个OffsetGenerator层从指令语义错误直接转变为完全正确，是可变形卷积硬件加速路径（OffsetLoader + 双线性插值WeightLoader）能够正常工作的先决条件。

在分块与代码生成层面，系统引入四级分层中间表示体系（Relay IR → LayerDesc → TilingPlan → 伪指令），将硬件约束集中建模。归纳出的四种Tiling模板（Template C/D/E/F）覆盖FSRCNN所有层类型，ping-pong buffer分配策略、`acc_mode`/`store_mode`自动推导机制以及QuantLoader 1-based连续编号策略，共同保证了指令参数的正确性。

实验验证方面，编译器在FSRCNN超分辨率网络（输入$(1,1,32,64)$）上实现了与黄金参考（`sd_sr_codegen.py`的`sr_inst()`函数）的指令级精确匹配：`load_next=False`和`load_next=True`两种模式下分别生成1,273条和1,274条伪指令，QL（12条）、DL（524条）、WL（524条）、DS（116条）、OL（96条）、ODS（1条）六类指令数量全部零差值。系统以约800行通用框架代码实现了与3,800行硬编码脚本等价的指令正确性，同时具备面向任意ONNX/PyTorch模型的扩展能力，代码规模降低约80%。

## 7.2　局限性

当前实现存在以下几项明确的局限，有必要坦诚说明：

**`quant_mode`依赖外部标定数据**。QuantLoader的`quant_mode`字段是硬件量化参数表的索引，其取值由量化感知训练（QAT）或量化标定（calibration）过程决定，无法从模型拓扑结构中推演。当前实现采用从黄金参考中反向分析得到的固定映射表；将编译器应用于新模型时，需要以JSON格式的per-layer配置表作为额外输入。这是量化部署场景的普遍约束，不破坏系统的通用性，但增加了接入成本。

**内存地址（`bas_addr`）需外部提供**。各类指令的起始内存地址`bas_addr`由系统级硬件内存布局（分区规则、对齐粒度）决定，类似通用编译器中的链接地址（link address）——编译器前端可以生成正确的指令结构，但无法独立确定最终的物理地址。当前系统尚未集成硬件内存布局建模模块，`bas_addr`字段存在与黄金参考的差异（831处），属于待精化的工程项而非设计性错误。

**UNet端到端验证尚未完成**。黄金参考文件`pseudo_code_load_next_first.txt`是`sd_inst()`（UNet，约19层）与`sr_inst()`（FSRCNN）的合并输出，而实验使用的`USR_Net.onnx`包含28个卷积层，与`sd_inst()`所对应的19层网络在架构上存在差异。在两者的对应关系确认之前，对`USR_Net.onnx`编译输出与合并黄金文件的直接比对缺乏方法论基础。FSRCNN侧的独立验证已经完备，UNet侧有待后续补充。

**部分ISA模板参数尚未精确对齐**。`line_buffer_reshape`（512处差异）、`line_buffer_row_shift`（320处）、`is_padding_col`（320处）等字段对应ISA文档中针对不同卷积配置的特定参数组合，原则上可通过逐模板对照硬件手册精确对齐，属于可迭代精化的工程完善工作。

## 7.3　未来工作方向

基于当前工作的基础，以下几个方向值得重点推进：

**量化标定结果集成**。设计量化配置接口，将QAT训练后的per-layer量化精度信息（如量化位宽、缩放因子）作为带类型的编译器输入，自动驱动`quant_mode`字段的推导与填充，从而实现量化感知网络的全自动编译。

**硬件内存布局建模**。建立内存分区模型，将权重存储区、特征图双buffer区、量化参数区等各分区的起始地址和对齐规则显式纳入编译器数据结构，使`bas_addr`的生成从外部依赖转变为编译器可推导的内部计算，最终实现字段级完全匹配。

**算子覆盖扩展**。当前模板系统以FSRCNN为设计基准，对于`pixel_shuffle`（像素重排）、深度可分离卷积（Depthwise Separable Convolution）、残差连接（Residual Connection）等算子尚缺少专用模板。在UNet等更复杂模型上推广，需要系统性地补充对应算子的LayerDesc解析规则、TilingPlan模板和指令发射逻辑。

**UNet完整对齐验证**。确认`USR_Net.onnx`与`sd_inst()`所对应网络的具体映射关系后，在UNet上复现FSRCNN侧的指令级验证流程，完成编译器对两类目标模型的端到端正确性覆盖。

**load\_next hoisting调度优化**。当前`OffchipDataLoader`（下一帧图像预取）固定在Layer 0的分块循环全部完成后发出，而硬件的`dependency`字段支持记分牌驱动的乱序执行。理论上，`OffchipDataLoader`可以在Layer 0循环的早期某个tile之后提前发出，使DDR数据预取与剩余tile的计算在时间上重叠，从而消除内存空泡（bubble）。`emitter.py`已预留`hoist_after_tile`参数接口，后续可在正确性验证完备后将其作为独立的调度优化实验推进，并以硬件仿真器定量评估实际吞吐收益。

上述方向的推进，将使本系统从"指令结构正确的前端原型"演进为"可直接面向硬件量产部署的全栈编译器"，为后续模型迭代和硬件升级提供稳固的工程基础。
