# 第一章　引言

## 1.1　研究背景与问题动机

深度卷积神经网络（Convolutional Neural Network, CNN）近年来在图像超分辨率（Super Resolution, SR）、目标检测、语义分割等视觉任务上取得了显著成效，这种广泛的成功持续推动着专用硬件加速器的需求增长。以FSRCNN[6]为代表的超分辨率网络要求在毫秒级延迟内完成数百万次乘累加运算，通用处理器在功耗与吞吐两个维度上均难以满足实时部署的要求。定制化神经网络加速器（Application-Specific Integrated Circuit, ASIC）则通过面向卷积算子的脉动阵列（Systolic Array）与层次化片上存储，在远低于GPU功耗的条件下提供数倍乃至数十倍的吞吐优势[11][12]。

硬件能力的跃升，并没有自动解决"如何让模型跑在新硬件上"这一工程难题。约3,800行Python脚本——参考实现`sd_sr_codegen.py`——清楚地说明了当前主流做法的局限：开发者逐层分析神经网络拓扑，手动计算分块参数（tiling parameters）、地址偏移与循环次数，再将这些常量硬编码（hardcoded）进面向特定硬件指令集（Instruction Set Architecture, ISA）的脚本中。该脚本专门针对FSRCNN和UNet两款固定架构，将每一条硬件伪指令（Pseudo-Instruction）以具体数值写死：一旦输入分辨率变化，或模型结构稍作调整，开发者就必须在整个文件中逐一定位散落各处的硬编码常量，逐条修改。

这种手写方式在模型种类少、硬件稳定的早期阶段尚可维持。代价随业务需求多样化而急剧攀升，问题最终收敛于三个维度。

**可维护性差**：指令参数与具体输入尺寸紧耦合，缺乏统一的分块抽象。这意味着任何改动都像是在一张未经整理的电路图上直接焊线——改动极易引入难以察觉的错误，且错误往往只在运行时才暴露。

**扩展代价高**：支持一个新模型几乎等同于从头重写整个脚本，工程量以周为单位计。而支持可变形卷积（Deformable Convolution）等非标准算子，更需要开发者深刻理解硬件单元协作语义，门槛高且错误风险大。

**非标准算子覆盖难**：目标加速器提供了可变形卷积的专用硬件路径（OffsetLoader + 双线性插值WeightLoader），但该算子在计算图层面由池化、卷积、双线性插值等多个基础算子组合而成，无法依靠简单的逐算子翻译实现正确的指令映射。

面对上述困境，以TVM[15]为代表的AI编译器（AI Compiler）提供了一条系统化的解决路径。其核心思想是：通过统一的中间表示（Intermediate Representation, IR）将模型导入与硬件代码生成解耦，用N+M的线性工程代价应对N个模型与M种硬件的适配问题[14]。然而，TVM的通用优化路径——无论是默认的算子融合规则还是代码降级（lowering）流程——均面向GPU/CPU设计，无法直接生成符合定制硬件ISA约束的正确指令序列。如何在复用TVM前端能力的同时，为特定加速器建立精确可控的指令生成链路，是将AI编译器落地于自研硬件的核心挑战。

## 1.2　相关工作的不足

AI编译器领域的进展相当可观。TVM通过Relay IR[16]提供了多框架模型导入和丰富的图级优化Pass，Ansor[19]自动化了张量程序（Tensor Program）的调度搜索，Halide[17]系统性地分离了算法与调度的表达，MLIR[18]以可扩展方言（dialect）机制构建了工业级的多层IR框架。这些工作共同推进了通用深度学习编译器的边界。

但当部署目标换成自研定制加速器，上述工作的局限便逐一浮现。

**自定义算子的完整支持缺失**：可变形卷积等非标准算子在通用IR中缺乏一等公民（first-class）语义。问题不止于此。`relay.frontend.from_pytorch`虽然内置了对`torchvision::deform_conv2d`的导入支持，将其表示为`nn.deformable_conv2d`之后，TVM的通用lowering路径随即将该算子展开为一系列基础索引操作和逐元素计算，绕过了目标加速器上OffsetLoader与双线性WeightLoader的专用硬件路径。更根本的困难在于：可变形卷积的偏移量生成子网络（OffsetGenerator）在通用IR中只是一个普通的`pool2d + conv2d`序列，其"输出应写入`offset_reg`而非feature buffer"的硬件语义在标准算子接口中没有任何对应表示——这一语义必须由编译器侧的结构性模式识别Pass来恢复。

**硬件ISA精确约束建模不足**：定制加速器的指令生成受到若干严格约束：分块粒度（如4行、32列、8通道）必须与片上buffer物理容量严格对齐；ping-pong（乒乓）双buffer的读写方向须逐层正确交替；QuantLoader的层索引必须连续且仅对特定算子类型递增。TVM的AutoTVM[21]和Ansor[19]面向GPU/CPU的调度空间（loop tiling、vectorization、unrolling），其代价模型（cost model）建立在通用计算平台性能模型之上，对"SRAM槽位整除约束"、"line buffer行容量上界"一类的硬件特有离散约束没有建模能力。MLIR的Affine Dialect[18]提供了仿射变换合法性检查框架，但其约束集针对连续仿射不等式，对本文目标加速器中以整除关系和物理容量限制为主的离散整数约束同样难以直接适用。

**量化感知网络到定制硬件的完整链路缺失**：从量化感知训练（Quantization-Aware Training, QAT）模型的导入，到包含量化参数（`quant_mode`）的硬件指令生成，再到多帧流水调度，这条链路上存在众多离散设计决策，现有工作没有系统性地解决过。TVM的Quantize Pass[16]针对的是软件量化——生成量化后的浮点或整数计算图——而非驱动具有独立硬件量化加速单元（QuantLoader + 片上量化参数表）的定制ASIC；ONNX Runtime的量化工具同样关注推理精度与存储压缩，不涉及特定硬件的`quant_mode`字段生成。结果是：即便上游量化工具链已输出per-layer的量化配置，现有编译器框架也无法将其自动转换为目标ASIC所需的QuantLoader指令参数，仍须人工介入建立映射关系。

**面向分组卷积和像素重排上采样的完整支持不足**：SD-UNet等视频超分辨率网络广泛采用分组卷积（Grouped Convolution）和DepthToSpace（像素重排）来实现高效的特征提取与上采样。分组卷积在硬件上以多轮QuantLoader+DataLoader+WeightLoader+DataStorer序列实现（而非单条分组卷积指令），循环层次（level1×level2）和QuantLoader发射时机随groups值和空间尺寸的变化而变化；DepthToSpace在目标加速器上不产生独立指令，而是通过前驱Conv层DataStorer的`is_pixelshuffle`等字段透明注入。这些规则高度特定于硬件架构，通用编译器框架不提供相应的代码生成路径。

综合来看，现有AI编译器工作在多框架模型导入和通用优化Pass层面已相当成熟，但在精确建模特定硬件ISA约束、将非标准算子子图语义正确映射到专用硬件单元、以及打通量化配置到指令生成的完整链路等关键环节上，仍存在显著的技术空白。本文以一款具体的自研CNN加速器为目标，设计并实现一套系统性的编译器前端解决方案，覆盖从多框架导入到精确指令生成的完整流程。

## 1.3　本文贡献

本文设计并实现了一个基于TVM Relay IR的编译器前端，专门面向自研CNN硬件加速器，将PyTorch/ONNX格式的神经网络模型自动转换为硬件伪指令序列，取代原有手写代码生成器。主要贡献如下：

**贡献一：多框架模型导入与统一IR表示**。我们设计了ONNX（`load_onnx`）和PyTorch（`load_pytorch`）双入口，以TVM Relay IR为统一中间表示，自动完成从计算图到分层中间描述（LayerDesc）的端到端解析。实现中发现并修复了TVM `relay.Expr`跨属性访问时Python包装对象哈希不稳定的缺陷（使用`expr in visited`替换`id(expr) in visited`），将编译时间从超时缩短至16毫秒。对于FSRCNN使用的`torchvision.ops.deform_conv2d`，我们发掘了TVM内置转换支持，实现了对可变形卷积的透明导入。

**贡献二：OffsetGenerator子图融合Pass**。我们提出`fuse_offset_generators` Pass，依据连续三层的结构性模式（`pool2d → conv2d(cout=18) → deformable_conv2d`），将偏移量生成子网络识别并融合为`offset_gen`算子，确保其DataStorer指令以`dest_buffer_idx='offset_reg'`写出，而非误写入普通数据buffer。该Pass直接决定了后续OffsetLoader的有效性：FSRCNN中全部4个OffsetGenerator层的指令生成从语义错误（0条`dest=offset_reg`的DataStorer）转变为完全正确。

**贡献三：分层中间表示体系与硬件约束建模**。四级分层表示体系——从Relay IR到LayerDesc、再到TilingPlan、最终到伪指令流——将硬件约束集中建模于各层之间的规范化接口，消除了手写方案中散布各处的硬编码常量；约束项涵盖分块参数模板、ping-pong buffer分配、多帧流水调度与QuantLoader连续编号策略。其中归纳出的五种Tiling模板（Template C/D/E/F及offset\_gen专用模板）覆盖FSRCNN全部12种计算层类型；面向SD-UNet引入形状键查表（`_UNET_LAYER_TABLE`，17条形状条目）配合idx二级消歧（`_UNET_IDX_OVERRIDE_TABLE`），以及`oc_inner`外层循环机制，支持编解码器对称网络的分块参数精确配置。

**贡献四：分组卷积与上采样算子的完整支持**。针对SD-UNet的分组卷积（Grouped Convolution），本文提出双级循环发射框架，覆盖groups=2（conv6，单级group循环）和groups=8（conv7/conv8单级外循环、conv10真双级嵌套）四种模式，QuantLoader发射时机由`group_ql_in_level2`标志统一控制。DepthToSpace采用透明化注入方案：在前驱Conv层DataStorer中注入`is_pixelshuffle`及相关字段，将像素重排上采样映射到硬件专用字段，不增减指令条数。pool-while-store机制则由三部分协同实现：pool2d层在IR中占位保留形状信息，Emitter层透明跳过，前驱Conv DataStorer中注入`is_pooling=1`字段，三者共同覆盖硬件池化语义。

**贡献五：两个目标网络的端到端指令级精确验证**。在FSRCNN超分辨率网络（输入$(1, 1, 36, 64)$，12层）上，与`sr_inst()`黄金参考的指令数完全对齐（`load_next=False`：1,273条；`load_next=True`：1,274条，六类指令数量逐项匹配）。在SD-UNet（USR\_Net\_109\_nopad.onnx，输入$(1,1,144,256)$，19层Conv，含groups=2/8、DepthToSpace×5、Concat×4）上，与`sd_inst()`黄金参考的指令总数精确匹配（17,155/17,155，功能性diff=0）；剩余14,664个字段差异经逐层multiset分析全部确认为非功能性（WL `is_new`顺序调度差异及QL寄存器编号差异）。编译器以约800行通用框架代码实现了两个网络的完整指令生成，具备面向任意ONNX/PyTorch模型的扩展能力。

## 1.4　论文组织结构

余下各章组织如下：

**第二章**回顾研究背景，介绍CNN与图像超分辨率的技术基础、专用深度学习加速器的架构设计，以及AI编译器领域的主要工作与挑战，为后续设计阐述建立技术语境。

**第三章**介绍编译器整体设计，涵盖TVM Relay IR的选型动机、ONNX/PyTorch双前端入口的实现细节、四级流水线架构，以及从Relay IR到LayerDesc的提取机制（含TVM哈希不稳定问题的发现与修复）。

**第四章**聚焦定制化算子支持，以可变形卷积为核心案例，描述硬件映射挑战、`torchvision.ops.deform_conv2d`的自动转换发现、专用指令发射模板的设计，以及`line_buffer_idx`不变式和QuantLoader连续编号策略的实现原理。

**第五章**阐述优化Pass的设计与收益，重点介绍OffsetGenerator子图融合Pass（含融合识别规则、专用TilingPlan参数和指令发射模板），以及Post-Pass的依赖分析、虚拟寄存器分配算法和多帧流水调度指令序列设计。

**第六章**报告实验与评估结果，对FSRCNN和SD-UNet进行指令级精确匹配验证，定量分析各优化Pass的贡献，重点阐述SD-UNet 14,664个字段差异的非功能性证明（逐层multiset分析方法），以及与手写代码生成器在代码规模、可维护性和通用性上的系统对比。

**第七章**从IR设计、分块调度、硬件接口正确性三个层面归纳技术贡献，讨论顺序调度限制、`quant_mode`依赖等当前局限性，并展望交错调度优化、量化标定集成、精确地址推导和MLIR演进等未来工作方向。
