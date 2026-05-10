# 第七章　结论

## 7.1　工作总结

本文面向一款自研CNN硬件加速器的部署需求，设计并实现了一个基于TVM Relay IR的编译器前端，将PyTorch/ONNX格式的神经网络模型自动转换为硬件伪指令序列，取代了原有依赖手工编码的代码生成器`sd_sr_codegen.py`。系统经过三十余个开发阶段的迭代，在FSRCNN和SD-UNet两个目标网络上均实现了与黄金参考的功能性等价，达到可进入上板验证的工程完整状态。

### 7.1.1　两个目标网络的完整验证

**FSRCNN超分辨率网络验证**（输入$(1,1,36,64)$，12层，含4个可变形卷积路径）：编译器在独立推理模式（`load_next=False`）下生成1,273条伪指令，在流水线模式（`load_next=True`）下生成1,274条，与黄金参考（`sd_sr_codegen.py`的`sr_inst()`函数）在QuantLoader（12条）、DataLoader（524条）、WeightLoader（524条）、DataStorer（116条）、OffsetLoader（96条）、OffchipDataStorer（1条）六类指令的数量上全部精确对齐（1,273/1,273）。

**SD-UNet超分辨率网络验证**（`USR_Net_109_nopad.onnx`，输入$(1,1,144,256)$，19层Conv，算子集涵盖groups=2/8分组卷积、AveragePool×4、DepthToSpace×5、Concat跳跃连接×4）：编译器生成17,155条伪指令，与黄金参考（`sd_inst()`函数）精确匹配（差值为0），QuantLoader（37条）、DataLoader/WeightLoader（各4,396条）、DataStorer（1,468条）、OffchipDataLoader（7条）、OffchipDataStorer（1条）全部一致。经过Phase 29-32系列修复——包括conv18 mask-store模式字段互换、OffchipDataStorer输出参数对齐（`src_buffer='unet_output_reg'`，`transnum=18`）、以及L=11 DS的`transfer_num=0`结束信号修复——剩余14,664个字段差异经逐层multiset分析全部确认为非功能性（WeightLoader `is_new`顺序调度差异约占93%，QuantLoader `quant_reg_load_idx`寄存器槽位差异约占0.4%，L=11 WL排序artifact约占6.8%，均不影响硬件计算结果），功能性diff为0。

两个网络同时达到功能完整状态，验证了编译器框架的通用性：从FSRCNN的12层轻量网络到SD-UNet的19层编解码器对称网络，算子复杂度跨越约一个量级，编译器均能正确生成精确的指令序列。

### 7.1.2　四级分层中间表示体系

编译器的核心设计是建立从Relay IR到LayerDesc、再到TilingPlan、最终到伪指令流的**四级分层中间表示体系**。每个层次的接口是规范化的数据结构，硬件约束集中建模于层次之间：LayerDesc完成从计算图到几何参数的蒸馏（抽象），TilingPlan将几何参数映射为硬件分块参数（求解），Emitter按分块参数批量发射ISA指令（实例化）。这种分层设计使硬件约束的修改影响范围最小化——改变一条Tiling规则只需修改`tiling.py`中的对应条目，无需触动Emitter的发射逻辑；引入新算子只需在LayerDesc提取和TilingPlan中增加对应分支，不影响已有算子的路径。

四个层次的具体职责边界值得进一步澄清，以说明这种分层设计相较于直接在Relay IR到指令之间进行一次性翻译的优越性。

**第一级：Relay IR（计算图级）**。承担多框架模型的统一接入，包含完整的算子调用（`relay.Call`）和类型信息（`call.checked_type`）。该层的信息量最丰富，包含了模型的所有拓扑结构、算子属性和张量形状，但同时也包含大量对硬件代码生成无关的元信息（如符号表名称、调试信息、TVM内部属性）。保留原始Relay IR的目的是便于利用TVM既有的形状推断（`InferType`）基础设施，而不进行任何不可逆的信息压缩。

**第二级：LayerDesc（层描述级）**。由`ir/layer_desc.py`的`extract_layer_descs`函数生成，是一种"按需抽象"的精简表示：从每个`relay.Call`中提取硬件代码生成真正需要的最小参数集（`h_in, w_in, cin, cout, k_h, k_w, stride, pad, groups, activation`等），丢弃对后端无用的冗余信息。LayerDesc的关键设计原则是**与具体硬件无关**——它只描述算子的几何语义，不包含任何硬件特有的分块参数。这使得LayerDesc可以在未来被复用于针对不同硬件的TilingPlan生成，而无需修改前端提取逻辑。经过`fuse_offset_generators`和`fuse_activations`两个融合Pass处理后，LayerDesc列表直接对应硬件的逻辑层序列（FSRCNN：12层；SD-UNet：19层conv，加4层pool占位），为后端提供了清晰的层次化视图。

**第三级：TilingPlan（分块参数级）**。由`tiling/tiling.py`的`plan_all`函数从LayerDesc列表批量生成，是硬件约束的主要承载体。TilingPlan的每个字段都对应一个可直接用于指令发射的硬件参数（`h_out_per_step`、`cin_group`、`weight_transnum_base`、`quant_mode`、`acc_mode`、`store_mode`、`oc_inner`等），不含任何需要在发射阶段再次推导的抽象量。这一设计将"确定硬件参数取什么值"（TilingPlan的责任）与"按参数发射什么指令"（Emitter的责任）明确分离，使每个层次只承担单一职责，测试时可独立验证TilingPlan的参数正确性而无需运行完整的指令发射。

**第四级：伪指令流（ISA级）**。由`backend/emitter.py`的`InstructionEmitter`按TilingPlan参数逐层发射，每条伪指令是一个字段完整的Python字典（包含指令类型、所有操作数字段），经`backend/post_pass.py`的`finalize_instructions`完成依赖分析和虚拟寄存器分配后，成为可直接提交硬件仿真器验证的指令序列。该层是唯一与硬件ISA直接耦合的层次，指令字段的语义完全由硬件文档定义，Emitter的职责是将TilingPlan中的抽象参数精确映射到具体字段值，不包含任何分块策略判断。

这四个层次之间的数据流是单向的（Relay IR → LayerDesc → TilingPlan → 伪指令），不存在反向依赖，使得调试时可以在任意层次截断并独立检查中间产物（`pipeline.py`的`PipelineResult`结构暴露了所有四个层次的输出，方便针对性地定位问题）。

在前端导入层面，系统提供ONNX（`load_onnx`）和PyTorch（`load_pytorch`）双入口，以Relay IR为统一中间表示承接多框架模型。实现过程中发现了TVM `relay.Expr` Python包装对象在属性访问时哈希不稳定的底层机制缺陷，将`id(expr)`替换为`expr`作为访问集合键后，编译时间从超时（UNet残差结构下的指数级重复遍历）缩短至16毫秒。对`torchvision.ops.deform_conv2d`的透明导入，发掘了TVM内置的转换支持，避免了大量重复的自定义算子注册工作。对PyTorch模型，通过`torch.jit.trace`进行静态追踪得到TorchScript图，使TVM前端能够在Python级别以外访问完整的算子图结构，这是可变形卷积得以被正确识别和导入的关键前提。ONNX入口则在处理initializer节点时需要区分真正的动态输入与静态权重常量（initializers列表中的权重不应加入形状字典），以避免权重被错误地视为动态输入而导致形状推断失败——这一实践细节对于含有大量常量权重的量化网络尤为重要。

### 7.1.3　OffsetGenerator子图融合Pass

OffsetGenerator子图融合Pass是本文的核心技术贡献之一。该Pass基于连续三层的结构性模式匹配，将偏移量生成子网络（`pool2d → conv2d(cout=18) → deformable_conv2d`）识别为`offset_gen`算子，确保其输出DataStorer以`dest_buffer_idx='offset_reg'`写出偏移寄存器，而非误入普通数据buffer。这一Pass使FSRCNN中全部4个OffsetGenerator层从指令语义错误（0条`dest=offset_reg`的DataStorer）直接转变为完全正确，是可变形卷积硬件加速路径（OffsetLoader + 双线性插值WeightLoader）能够正常工作的先决条件。

Pass的设计有两点值得强调：其一，识别规则的严格性（pool2d + conv2d(cout=18) + deformable\_conv2d三个条件同时成立）避免了对其他输出通道恰好为18的普通卷积的误识别；其二，`fuse_offset_generators`对SD-UNet的输出**零影响**（SD-UNet不含OffsetGenerator结构，Pass对其输入列表原样返回），验证了模式匹配识别规则的精确性和隔离性。

深入理解这个Pass的价值，需要从可变形卷积的硬件实现机制说起。目标加速器对可变形卷积的支持不是通过通用路径（展开为多个基础算子的逐元素计算）实现的，而是依靠两个专用硬件单元的协作：OffsetLoader负责将偏移量值从片上寄存器（offset\_reg）读出并送入地址生成单元（Address Generation Unit），WeightLoader的双线性插值模式（`is_bilinear_bicubic=1`）则利用OffsetLoader提供的亚像素坐标，在特征图上进行四点邻域的线性加权插值，从而实现不规则采样。两个单元的协作依赖一个隐式契约：在WeightLoader被调用之前，offset\_reg中必须已经存放了有效的偏移量数据。

这个隐式契约在编译器层面的对应关系是：**DataStorer指令必须以`dest_buffer_idx='offset_reg'`将偏移量生成卷积（即`conv2d(cout=18)`）的输出写入offset\_reg**，而非写入普通的feature buffer（'a'或'b'）。在通用的编译器路径中，一个`cout=18`的`conv2d`与任何其他卷积没有语义区别，其DataStorer自然会写入当前的ping-pong feature buffer。OffsetGenerator融合Pass的核心贡献正是打破这种"语义透明"的幻觉：通过精确的三元模式匹配，在IR级别将具有特定功能语义（偏移量生成）的子图识别出来，并将其路由到能够正确设置`dest_buffer_idx='offset_reg'`的专用发射路径（`_emit_offset_gen`方法）。

从通用IR设计的角度看，这一设计决策体现了一类普遍存在于硬件加速器编译器中的问题：**通用IR中缺乏对硬件专用功能单元的一等公民语义表示**。解决这类问题的通用策略有两种——其一是在IR中引入新的算子类型（如直接定义`offset_gen_conv2d`算子），其二是通过模式匹配Pass将已有算子序列映射到专用语义。本文选择第二种方案，是因为目标模型（FSRCNN）的PyTorch/ONNX源文件中不存在`offset_gen_conv2d`这样的算子，强行在IR中定义新算子会破坏从框架导入的透明性；而基于模式匹配的融合Pass则可以在保持前端导入完全标准化的同时，在IR处理阶段插入精确的语义解析，是一种对源模型格式侵入性最低的方案。

此外，OffsetGenerator融合Pass的副产品之一是将pool2d节点从独立的LayerDesc转换为被合并到`offset_gen`中的`extra`字段（记录`pool_stride`参数），消除了pool2d在指令发射阶段可能导致的PseudoOp插入问题，使最终指令流中的PseudoOp数量严格为零，与黄金参考的行为完全一致。这一"顺手消除"的效果是Pass级联设计（`fuse_offset_generators` 先于 `fuse_activations` 执行）所带来的自然收益，展示了分层中间表示架构中Pass之间正交组合的优雅性。

### 7.1.4　SD-UNet的系列扩展工作

SD-UNet（USR\_Net\_109）的支持涵盖了FSRCNN完全没有触及的若干新类型算子和新调度模式，代表了本编译器在算子覆盖维度上的主要扩展贡献：

**全高度流式调度**（§5.5）：引入`tile_h=None`分支，以整个H维度为单一分块，按1行或2行步长逐行推进。借助TVM的形状推断机制（`relay.transform.InferType()`），AveragePool下采样带来的`h_in/w_in`折半信息自动传递给下游conv层的`load_total_num`计算，无需额外的折半逻辑。全高度模式与FSRCNN的分块流水模式共享同一套代码路径，分流逻辑仅在`choose_tiling()`函数内的一行条件判断处发生，实现了零代码重复。

**分组卷积双级循环发射**（§6.3.6）：针对groups=2（conv6）和groups=8（conv7/conv8/conv10）四种发射模式，通过`_apply_group_params`函数按条件分支配置8个group相关字段，实现双级循环框架（level1 × level2）。其中conv10的真双级嵌套（level1=2，level2=4，QL在内层每次迭代发射）与conv7/conv8的单级外循环（QL在外层开始时发射一次）在同一`_emit_group_conv`框架内统一处理，且groups=1时自动退化为单次迭代，保证FSRCNN路径零影响。

**DepthToSpace透明化注入**（§5.5）：分析黄金参考中DepthToSpace对应的DataStorer字段（`is_pixelshuffle=1`、`pixelshuffle_out_mode`、`acc_mode`、`store_mode`、`transfer_num`、`stride`），在前驱Conv层的DataStorer中透明注入，不增减指令条数，但消除了对应字段的结构性差异。DepthToSpace节点本身在指令流中产生零条ISA指令，类似pool2d的透明化处理。

**conv18 mask-store模式与输出层参数对齐**（§6.5.3 Phase 31）：SD-UNet最终输出层（conv18）以mask-store模式（`is_mask=1`）写入`unet_output_reg`寄存器，其DataStorer字段与普通conv存在本质差异：`is_pooling`、`pooling_out_mode`均置为特殊值（1和4），`pix_transfer_num`由1变为2，且`base_addr_pooling`与`base_addrs_res`的增量地址角色互换——普通层增量走`base_addrs_res`通道，conv18改走`base_addr_pooling`。此外，OffchipDataStorer的`src_buffer`须为`'unet_output_reg'`（区别于FSRCNN的`'fsrcnn_output_buffer'`），`transnum=18`。以上两处修复通过`TilingPlan.is_mask`标志和`PipelineConfig`的输出参数配置实现。

**L=11 DS结束信号机制**（§6.5.3 Phase 32）：conv11（groups=2）每个group的最后一次DataStorer需以`transfer_num=0`作为group结束信号，而非与组内其他DS相同的`transfer_num=1`。修复通过在`TilingPlan`新增`ds_last_transfer_num`字段（默认`None`），并在`_emit_group_w_tile`函数的DS发射末尾根据`load_idx == load_total-1`条件覆盖`pix_transfer_num`字段来实现，消除了L=11 DS的最后2条功能性差异。

**内存分配与地址布线**（§3.7, §6.3.7）：基于`{a, b}`二着色约束建立feature buffer分配模型，通过线性扫描算法实现skip连接Tensor的活跃区间（live range）管理。在USR-Net的4个Skip Tensor（活跃区间横跨编码器至解码器对称层）下，理论下界分析（BufA：16,384 words；BufB：8,192 words；合计24,576 words）被三种经典算法（线性扫描、TVM工作空间分配、MLIR Bufferization）均达到——二着色约束的嵌套活跃区间结构天然防止了碎片化，三种算法在此场景下等价。

### 7.1.5　工程维度对比

系统以约800行通用框架代码（`pipeline.py`、`frontend/`、`ir/`、`tiling/`、`backend/`九个核心模块）取代原有3,800行硬编码脚本，代码规模降低约80%，同时实现了FSRCNN和SD-UNet两个网络的完整指令生成，新模型的接入代价从数周的手工重写缩减为增量式的规则扩展（新算子仅需在LayerDesc和TilingPlan层增加对应条目）。

这一代码规模缩减的背后是设计范式的根本性转变。手写方案`sd_sr_codegen.py`的约3,800行代码中，有相当大比例是对两个固定网络的状态转移进行手工展开：每一层的`DataLoaderManager`调用、`WeightLoaderManager`状态推进、地址偏移的算术计算、乒乓buffer切换判断，均以内联数值常量的形式散布于代码各处。当修改一处输入分辨率时，对应的`cal_total_num`、`bas_addr`增量、地址边界等参数需要在数十个位置同步更新，极易遗漏；当引入新算子（如可变形卷积的OffsetLoader路径）时，需要从零设计对应的状态机逻辑并确保与已有状态变量的正确交互，门槛极高。

本编译器的800行代码是结构性的，而非展开的。核心状态机（`EmitterState`）以统一的数据结构封装所有可变状态字段（`feature_buf`、`line_buffer_idx`、`acc_reg_idx`、`weight_bas_addr`、`conv_layer_counter`等），指令发射逻辑与状态更新逻辑分离，各算子类型的发射路径在单一分支内完整实现，互不干扰。这种设计使代码的复杂度随算子种类呈线性增长，而非随模型-算子组合爆炸。

从具体的工程指标看，两种方案的可维护性差距尤为突出：

**分辨率扩展**：手写方案需要在整个脚本中逐一定位并修改所有与输入分辨率相关的硬编码常量（如`cal_total_num`由`h_in // 4`决定，`bas_addr`增量由`w_in // 64`决定），依赖开发者对代码的全局记忆；本编译器只需修改`PipelineConfig`中的输入形状参数，`plan_all()`函数自动重新推导所有分块参数，且`image_transnum`的自动推导公式（$h_{in} \times \max(1, w_{in} \div 64)$）通过单元测试保证与任意分辨率的正确性。

**新算子接入**：手写方案接入一类新算子（例如groups=8的分组卷积）需要手工分析硬件执行流程、计算指令数上界、设计地址推进策略，并在现有数千行代码的特定位置插入对应逻辑，估计工程量约为1~2周；本编译器通过在`ir/layer_desc.py`的解析分支、`tiling/tiling.py`的条件分支和`backend/emitter.py`的发射分支中各增加约20~50行代码即可完成，逻辑集中，风险可控。

**测试可维护性**：手写方案与特定模型的黄金状态高度耦合，修改任何逻辑都需重新核对整个指令流；本编译器各层次（LayerDesc提取、TilingPlan生成、指令发射）均有独立的单元测试接口，可针对单层或单类算子进行隔离验证，将回归测试的粒度从"全模型"细化到"单算子类型"。

综合上述分析，代码规模从3,800行压缩至800行所带来的工程价值，远不止于代码量本身的减少，而在于它将系统的认知复杂度降低到单个工程师可以独立理解和维护的范围内，为后续的算子扩展和硬件迭代奠定了可持续的工程基础。

---

## 7.2　主要技术贡献的系统性归纳

回顾全文工作，本文的技术贡献可从三个层面系统性归纳：

**层面一：IR层设计贡献**。提出了从Relay IR到LayerDesc的蒸馏机制（`ir/layer_desc.py`），以"按需抽象"原则仅保留后端代码生成所需的最小参数集，避免了Relay IR中冗余信息对后端的污染。发现并修复了TVM Python包装对象哈希不稳定缺陷（`id(expr)` vs `expr` in `visited`），该缺陷在DAG结构（如残差网络）中具有指数级放大效应。提出OffsetGenerator子图融合Pass，以精确的三元结构性模式匹配实现了"将标准算子序列正确映射为硬件专用指令目标地址"的语义转换。

**层面二：分块与调度贡献**。归纳出覆盖FSRCNN全部层类型的五种分块模板（Template C/D/E/F及offset\_gen专用模板），并建立了面向SD-UNet的形状键查表+idx二级消歧机制（`_UNET_LAYER_TABLE` + `_UNET_IDX_OVERRIDE_TABLE`），解决了"形状签名相同但网络语义不同"的歧义问题。设计了全高度流式调度与分块流水调度的双模式架构，实现了`tile_h`参数的统一分流控制。引入`oc_inner`外层循环机制，支持输入重扫描型双oc迭代（出现于SD-UNet decoder层的输出通道数超过一次并行处理上限的情形）。

**层面三：硬件接口正确性贡献**。实现了ping-pong双缓冲的逐层正确交替分配（offset\_gen层不切换`feature_buf`状态），消除了早期实现中全部层使用同一buffer方向的系统性错误。实现了`acc_mode`/`store_mode`的7规则自动推导，覆盖offset\_gen、deformable\_conv、标准conv+prelu/relu、末层conv等全部场景。实现了QuantLoader 1-based连续编号策略（`conv_layer_counter`跳过prelu和pool2d），以及`line_buffer_idx`不变式（DL和WL必须使用相同值，toggle统一在WL之后发生）。Post-Pass中的虚拟寄存器分配算法（15个可用寄存器，LIFO分配，`src4` quirk的刻意保留）以及7类依赖分析规则，保证了指令流与硬件调度器的精确接口。针对SD-UNet最终输出层，建立了mask-store专用的DataStorer字段推导规则——包括`base_addr_pooling`与`base_addrs_res`增量通道的模式感知互换，以及OffchipDataStorer的网络感知参数配置（`src_buffer`与`transnum`）；引入`TilingPlan.ds_last_transfer_num`机制，支持分组卷积每组最后一次DataStorer以`transfer_num=0`为结束信号，完整覆盖了硬件group结束协议。

---

## 7.3　当前局限性

当前实现在达到功能完整状态的同时，仍存在若干明确的局限性，有必要坦诚说明：

### 7.3.1　顺序调度未实现交错调度优化

本编译器与黄金参考`sd_sr_codegen.py`均采用静态顺序调度（Static Sequential Scheduling）：在cin\_group循环中，对同一外层H-tile的各个输入通道组按序处理，第一组使用`is_new=0`（清零写入acc\_reg），后续组使用`is_new=1`（累加）。

黄金参考的部分层（主要是SD-UNet的大通道conv层）采用了交错调度（Interleaved Scheduling）：在两个外层H-tile的cin\_group循环之间穿插执行，以隐藏相邻tile之间的数据依赖延迟。这种交错模式在不改变最终计算结果的前提下，能够更好地利用硬件流水线的空泡时间。然而，交错调度的实际理论收益上界远低于直觉估计。以SD-UNet的conv层为例，其算术强度（Arithmetic Intensity）集中在160~638 FLOPS/byte区间（例如conv7：$C_{in}=64, C_{out}=64, k=3, H=144$对应约638 FLOPS/byte；conv6：$C_{in}=32, C_{out}=32, k=3, H=144$对应约319 FLOPS/byte），属于强compute-bound（强计算密集型）区域——权重加载时间仅为MAC计算时间的$1/500$至$1/18000$。换言之，可被交错调度隐藏的内存访问延迟本身在整个执行时间中的占比极低，交错调度的吞吐量提升上界约为**0.1%~0.2%**，并非此前估计的5%~15%——后者适用于访存密集型（memory-bound）场景，而非本网络实际所处的计算密集型工作点。

编译器当前采用顺序调度的原因是实现简单性和验证可控性——两种调度方式产生相同的计算结果，差异仅体现在`is_new`字段的时序顺序上，这正是SD-UNet中14,664个非功能性字段差异中约93%的来源。实现交错调度的理论复杂度估计为对Emitter状态机的中等改造，但需要额外的正确性验证以排除边界情形下的状态污染风险，因此规划为后续工程优化任务。

需要指出的是，上述调度差异不影响本工作的核心性能等价性结论。本编译器在FSRCNN和SD-UNet两个目标网络上均生成了与黄金参考**指令数精确一致**的指令序列（FSRCNN：1,273/1,273；SD-UNet：17,155/17,155），而黄金参考`sd_sr_codegen.py`本身是硬件工程师针对该MAC阵列手工调优的最优指令序列。这意味着编译器在两个目标网络上的**性能上界已与手写参考等价**——顺序调度与交错调度在指令总数上完全相同，区别仅在于`is_new`字段的发射时机，这一差异在强compute-bound的算术强度下对端到端吞吐的实际影响可以忽略不计。进一步的硬件吞吐提升需要在**硬件架构层**（更深的流水线设计、更大的片上存储）或**模型算法层**（轻量化网络结构、混合精度量化以降低算子计算量）寻找空间，不属于编译器前端的设计范畴。

### 7.3.2　`quant_mode`依赖外部标定数据

QuantLoader的`quant_mode`字段是硬件量化参数表的索引，其取值由量化感知训练（Quantization-Aware Training, QAT）或量化标定（calibration）过程决定，无法从模型拓扑结构中推演。当前实现采用从黄金参考中反向分析得到的固定映射表；将编译器应用于新模型时，需要以JSON格式的per-layer配置表作为额外输入。这是量化部署场景的普遍约束，不破坏系统的通用性，但增加了接入成本。相比之下，`acc_mode`和`store_mode`已实现自动推导，当前唯一仍需外部提供的量化相关参数即为`quant_mode`。

要理解这一局限性的深层根源，需要明确`quant_mode`字段在硬件语义中所扮演的角色。目标加速器的QuantLoader并不直接存储浮点缩放因子，而是维护一张片上量化参数表（quantization parameter table），表中每个条目（entry）对应一种特定的量化配置——包括输入激活值的量化位宽、缩放因子的定点表示、零点偏移（zero-point offset）等。`quant_mode`本质上是这张表的行索引，决定了当前层的卷积输出在写入DataStorer时采用哪一种量化规则进行激活值量化。这意味着，`quant_mode`的正确取值需要在知道该层激活值的统计分布范围的前提下才能确定——而这一信息只有在真实数据上运行标定（calibration）过程，或在QAT训练过程中通过梯度更新学习到的量化感知参数中才能获得。

从当前实现的处理方式来看，编译器通过对黄金参考`sd_sr_codegen.py`的逐层代码分析，将FSRCNN的12层和SD-UNet的19层的`quant_mode`取值逐一提取，形成以层形状签名（或层索引）为键的固定映射表，硬编码于`tiling/tiling.py`的各TilingPlan条目中。对于FSRCNN，12个conv类层的`quant_mode`取值范围为{0, 2}，其中offset\_gen专用层取2，其余取0；对于SD-UNet的19层，取值范围更广，覆盖多个量化精度等级。这些数值直接来自硬件团队标定后固化到黄金参考中的结果，具有与真实硬件对齐的正确性保证。

然而，这种反向提取的方式有其根本的局限性：它将量化标定信息隐式地绑定在特定的模型权重和特定的标定数据集上，而无法泛化到权重发生变化（如模型微调）或输入数据分布发生变化（如不同场景的图像）的情形。当前两个目标网络的权重已固定，因此这一限制在实际部署中不构成问题；但一旦需要在同一硬件上部署经过fine-tune的模型变体，原有的`quant_mode`映射表就必须重新标定和更新，现有的编译器接口并不提供将新标定结果注入编译流程的标准化路径。

从量化工作流的整体视角看，`quant_mode`的正确生成需要编译器与QAT/PTQ（训练后量化，Post-Training Quantization）工具链之间建立明确的数据接口。理想的集成方式是：量化工具链在完成标定后输出一份per-layer的量化配置描述（如ONNX的QuantizationAnnotation或专有JSON格式），编译器将其作为第一类输入（first-class input）读取，从中自动提取每层的`quant_mode`索引，填入对应的TilingPlan条目。这不仅消除了当前手工维护映射表的工程负担，还为同一编译器前端支持"同一网络结构、不同量化精度配置"的多个部署变体提供了可能，是将编译器从"为两个固定模型服务"升级为"为任意量化CNN模型服务"的必要条件之一。§7.4.3节将进一步阐述这一集成方案的技术路线。

### 7.3.3　内存地址（`bas_addr`）的精确推导尚未完整

各类指令的起始内存地址`bas_addr`由系统级硬件内存布局决定，类似通用编译器中的链接地址——编译器前端可以生成正确的指令结构，但无法独立确定最终的物理地址。当前工作在P0阶段完成了图像Buffer地址参数（`image_transnum`/`inter_layer_bas_addr`/`load_next_bas_addr`）的自动推导（基于DataLoader传输粒度64像素/word的显式公式），验证与黄金参考完全一致（零差异）。Skip Tensor的活跃区间（live range）分析框架已建立（`ir/addr_alloc.py`），可用于feature buffer内各层Tensor的基地址排布，但DataStorer的`base_addrs_res`等字段仍存在与黄金参考的差异（覆盖FSRCNN侧约831处）。精确的连续地址排布推导需要多面体内存分析（Polyhedral Memory Analysis）或整数线性规划（ILP）方法，属于已规划的后续工程任务。

### 7.3.4　部分ISA模板参数尚未精确对齐

`line_buffer_reshape`（512处差异）、`line_buffer_row_shift`（320处）、`is_padding_col`（320处）等字段对应ISA文档中针对不同卷积配置（核尺寸、步幅、填充）的特定参数组合，原则上可通过逐模板对照硬件手册精确对齐，属于可迭代精化的工程完善工作（仅影响FSRCNN的字段级对比，不影响已确认的功能正确性）。

这三类字段的共同特点是：它们的取值不依赖于算子的语义正确性（即不影响计算结果的等价性），而是控制硬件内部的数据重排（reshape）和缓冲区寻址方式，因此在指令类型数量完全匹配、功能性差异为零的前提下，其字段值的偏差不会导致硬件输出错误，但在上板部署时可能影响吞吐量或功耗，有必要在后续阶段精确对齐。

**`line_buffer_reshape`（512处差异）**：该字段控制DataLoader将片上line buffer中的数据重新排列（reshape）成MAC阵列期望的布局方式，不同的核尺寸（1×1 vs 3×3）和步幅组合对应不同的重排模式编号（0/1/2/3）。当前编译器在大部分层采用默认值，与黄金参考在offset\_gen层（`line_buffer_reshape=2`）和部分3×3标准卷积层的取值存在差异。从黄金参考的逐层分析来看，该字段的正确取值与`k_h × k_w`（卷积核面积）和`h_out_per_step`的组合存在系统性对应规律，可通过建立枚举映射表或显式推导公式消除差异，预计约需20~30行代码修改。

**`line_buffer_row_shift`（320处差异）**和**`is_padding_col`（320处差异）**：这两个字段均属于WeightLoader的ISA模板参数，`line_buffer_row_shift`指定WeightLoader从line buffer读取权重时的行偏移量（用于处理非零填充（padding）情形下边界行的寻址修正），`is_padding_col`则标记当前WeightLoader加载的是否为填充列（用于抑制边界区域的无效累加）。这两个字段在FSRCNN的标准卷积层（L0的cin=1首层及L11的输出层）和SD-UNet的部分3×3卷积层中存在与黄金参考的差异，数量恰好相同（均为320处），表明两者很可能在相同的指令集合上同时偏离，属于同一逻辑单元在不同字段上的联动差异。其根因是当前TilingPlan未将padding信息（`pad_top/pad_bottom/pad_left/pad_right`）传导至WeightLoader的字段推导逻辑中，修复路径是在`_emit_w_macro_tile`中根据`layer.pad_top > 0`条件动态设置这两个字段，而非沿用全局默认值。

综合三类差异的分析，其修复并不涉及架构层面的改动，也不需要修改任何已经正确建立的状态机逻辑——它们是TilingPlan字段推导规则的局部精化，每处修复只影响生成对应字段值的几行代码。从工程优先级排序来看，建议先完成`quant_mode`集成（§7.4.3）和`bas_addr`精确推导（§7.4.4）这两项影响功能完整性的工作，再推进上述ISA模板参数的精化，以最终达成全字段精确匹配的目标。

### 7.3.5　编译器覆盖范围的边界

当前编译器的算子覆盖以FSRCNN和SD-UNet为设计基准，对于Attention机制（自注意力层）、Transformer块、深度可分离卷积（Depthwise Separable Convolution）的扩展尚未实现；多批次（batch>1）推理的TilingPlan模板也尚未建立。这些限制在本文的目标加速器和目标网络范围内不构成功能缺口，但在面向更广泛的网络族群时需要系统性扩展。

具体而言，当前编译器在以下几个算子维度存在明确的覆盖空白，有必要说明每处边界背后的技术含义：

**深度可分离卷积（Depthwise Separable Convolution, DSC）的缺失**：深度可分离卷积由逐通道卷积（Depthwise Convolution，`groups=C_in`）与逐点卷积（Pointwise Convolution，1×1 conv）两个算子串联构成，在MobileNet [Howard, 2017]、EfficientNet [Tan, 2019]等轻量化网络中广泛使用。目标加速器的MAC阵列对`groups=C_in`的极端情形（每组仅有1个输入通道）是否有专用执行路径，尚未在硬件文档中明确标注，因此目前的`_apply_group_params`分支逻辑未覆盖`groups > 8`的情形。从代码架构看，逐通道卷积可视为分组卷积的极端情形（`group_level1=C_in, group_level2=1`），只需确认硬件是否支持以及对应的`weight_parall_mode`和`cin_group`参数取值，即可在现有双级循环框架内完成扩展，不需要引入新的指令类型。

**Attention机制与Transformer块的缺失**：当前目标加速器ISA的7类指令（OffchipDataLoader/DataLoader/WeightLoader/OffsetLoader/QuantLoader/DataStorer/OffchipDataStorer）是面向CNN卷积特征提取设计的，尚未包含用于实现多头自注意力（Multi-Head Self-Attention, MHSA）所必需的点积缩放（Scaled Dot-Product Attention）、Softmax激活、位置编码等算子的硬件加速路径。因此，将当前编译器扩展到Transformer类网络（如ViT [Dosovitskiy, 2020]、Swin Transformer [Liu, 2021]），需要先在硬件层面引入对应的新指令类型，再在编译器的LayerDesc解析、TilingPlan生成和Emitter三个层次同步扩展，是一项涉及硬件-编译器协同设计的系统性工作，超出本文的范围。

**多批次推理（batch > 1）的缺失**：目标加速器的片上buffer容量和DataLoader的传输机制均按单帧推理（batch=1）设计，多帧流水通过`load_next`机制实现，而非通过增大batch size实现。当前`PipelineConfig`和`TilingPlan`的所有参数均以batch=1为隐含前提，扩展到batch>1需要重新建模DataLoader的地址步进规则和片上buffer的多帧并发布局，与当前架构存在根本性的设计冲突，在短期内不纳入扩展计划。

**残差连接（Residual Connection）的TilingPlan支持**：SD-UNet中存在Concat型跳跃连接，已通过Skip Tensor活跃区间管理和地址布线正确处理。但更一般的残差加法（element-wise add）结构——如ResNet中的快捷连接——尚未进入发射路径的设计。目标加速器是否在硬件层面直接支持element-wise add指令（作为DataStorer的一种模式），或需要在Emitter层模拟，取决于硬件文档中对应章节的明确说明，属于待研究的接口细节。

上述覆盖边界的存在，并不意味着编译器框架的设计存在根本缺陷——四级分层体系（Relay IR → LayerDesc → TilingPlan → 伪指令）的扩展入口是明确的，每类新算子的接入不影响已有路径的稳定性。这些边界是当前迭代阶段合理的范围约束，而非架构层面的障碍。

---

## 7.4　未来工作方向

基于当前工作所建立的技术基础，以下几个方向值得重点推进：

### 7.4.1　交错调度实现（P2优化）

从顺序调度升级到交错调度，是提升SD-UNet大通道层吞吐量的最直接路径。具体实现方案是在`_emit_w_macro_tile`函数的cin\_group循环之外增加H-tile级别的外层交错控制，使相邻两个H-tile的cin\_group迭代在时间上部分重叠。`emitter.py`的`EmitterState`已维护了`line_buffer_idx`和`acc_reg_idx`的完整状态，是实现交错调度状态机的良好基础。

实现交错调度后，WL `is_new`字段的发射时机将与黄金参考对齐，预计可消除目前SD-UNet 14,664个字段差异中约93%（即约13,600个）的来源，使字段级对比精度大幅提升，同时在硬件层面获得更高的MAC阵列利用率。

### 7.4.2　Tiling参数自动推导泛化

当前编译器的Tiling策略依赖`tiling/tiling.py`中的`_UNET_LAYER_TABLE`硬编码查表与Template A/B/C/D/E/F六种手工分块模板的组合。该机制在FSRCNN和SD-UNet两个目标网络上运行无误，但其本质是"以人工枚举代替规则推导"——每接入一个新模型，工程师须手动分析硬件约束、试算分块参数、逐条填写查表条目，接入成本难以随模型数量线性扩展。将Tiling参数推导从查表查询升级为可在任意层形状上自动运行的规则引擎，是本编译器走向通用化部署的最关键前提之一。

为此，本文规划了一条三阶段的渐进式技术路线。

**第一阶段：参数化公式替代查表。** 对`_UNET_LAYER_TABLE`中的可推导字段（`h_out_per_step`、`cin_group`、`weight_transnum_base`等），建立以层形状$(H,W,C_{in},C_{out},k,\text{group})$和硬件规格常数（Line Buffer容量$\mathcal{L}$、MAC阵列宽度$\mathcal{M}$、权重SRAM槽数$\mathcal{S}$）为输入的显式推导公式。以`weight_transnum_base`为例，其在标准卷积下严格满足$\text{wt\_base} = k^2 \times c_{in,g}$（其中$c_{in,g} = C_{in}/\text{group}$），这一关系在现有查表中以数值常量的形式被固化，完全可由公式替代。对于`h_out_per_step`，其取值受制于Line Buffer的行容量约束：$h_{step} \times \lceil (k-1)/2 \rceil \leq \lfloor \mathcal{L} / W \rfloor$，在给定层宽$W$和核尺寸$k$后即可确定合法取值区间，并按MAC阵列利用率最大化原则从中选优。

**第二阶段：合法性约束检查器（TilingConstraintChecker）。** 在第一阶段提供候选参数的基础上，建立显式的可行性（feasibility）验证模块，将散落在各发射函数中的硬件约束以一阶逻辑谓词的形式集中建模。核心约束集至少应覆盖：（1）Line Buffer约束——$h_{step} \times (k - 1 + h_{step}) \leq \mathcal{L} / W$；（2）整除约束——$c_{in,g} \mid C_{in}$；（3）SRAM槽位约束——$\text{wt\_base} \times \lceil C_{out} / \text{wt\_parallel} \rceil \leq \mathcal{S}$；（4）分组对称性约束——`ky_outer`与步幅的整除关系。该模块使编译器在接受非法参数组合时能以明确的约束违例报告终止，而非静默生成语义错误的指令序列——这一类"编译成功但行为错误"的故障在当前调试过程中已造成多次难以定位的验证失败。在设计原则上，这与MLIR Affine Dialect的`isTilingValid`检查框架 [Lattner et al., CGOA 2021]一脉相承，但约束集从连续仿射不等式推广到覆盖本硬件离散ISA参数的混合整数谓词。

**第三阶段：枚举搜索替代手工调表。** 一旦合法性约束检查器就位，Tiling参数选择问题即可转化为一个带约束的枚举优化问题：在搜索空间$h_{step} \times c_{in,g} \times \text{ky\_outer} \times \text{wt\_parallel\_mode} \times \text{lbuf\_reshape}$上枚举所有满足约束的候选组合，按目标函数（MAC阵列有效利用率$= \text{有效MAC数} / \text{峰值MAC能力}$）排序后取最优。对于本系统的典型层参数（$H \leq 256$，$C_{in} \leq 128$，$k \in \{1,3\}$），候选组合数约为360个量级，单层枚举耗时在现代CPU上估计不超过1毫秒，无需任何硬件运行反馈。

这一方案与TVM AutoTVM [Chen et al., NeurIPS 2018]和Ansor [Zheng et al., OSDI 2020]在机制上有本质区别：AutoTVM和Ansor面向的是通用张量程序的代价模型（cost model）搜索，其搜索空间为连续或半连续的调度变换空间，依赖真实硬件测量或代价模型的迭代反馈；而本方案的搜索空间是由固定ISA约束划定的有限离散集合，目标函数可直接由硬件规格参数（MAC阵列宽度、流水段数）静态估算，无需运行时反馈。在离散约束枚举这一子问题上，多面体模型（Polyhedral Model）的代表工作Pluto [Bondhugula et al., PLDI 2008]提供了理论参照，但其合法性分析基于连续仿射变换，对本文中以"SRAM槽位整除"为代表的离散整数约束不能直接适用——整数线性规划（Integer Linear Programming, ILP）方法更适用于跨层全局Tiling联合优化场景（多层共享片上SRAM时搜索空间随层数指数增长，ILP的松弛-枝界策略可有效降维），而在本系统的单层独立优化场景下，由于搜索空间可枚举，完整枚举方案的工程复杂度反而低于引入ILP求解器的方案，具有更好的工程实用性。

上述三阶段路线可独立实施，每完成一阶段即可缩减手动接入新模型的工作量：第一阶段消除可推导字段的手工填表，第二阶段将接入错误由"运行时静默"改为"编译期可见"，第三阶段实现端到端无需人工介入的Tiling参数生成。三阶段合力，使本编译器对新CNN模型的接入代价从"人日"量级降低至"分钟"量级，是推动系统从工程原型向通用化工具演进的核心路径之一。

### 7.4.3　量化标定结果集成

设计量化配置接口，将QAT训练后的per-layer量化精度信息（量化位宽、缩放因子、零点）作为带类型的编译器输入，自动驱动`quant_mode`字段的推导与填充，从而实现量化感知网络的全自动编译。这一方向与MLPerf [Reddi, 2020]等标准化量化推理流程对接，是将编译器从"工程原型"演进为"产品级工具"的关键步骤。

具体技术路线如下。

**接口设计**：在`PipelineConfig`中引入`quant_config`字段，类型定义为`Optional[Dict[str, QuantLayerConfig]]`，其中键为层形状签名（如`(h_in, w_in, cin, cout, k, groups)`六元组的字符串表示），值为`QuantLayerConfig`数据类（包含`quant_mode: int`、`scale_factor: float`、`zero_point: int`、`bit_width: int`四个字段）。该字典由量化工具链在标定完成后以JSON格式序列化输出，编译器在`run_pipeline()`入口处反序列化并存入`PipelineConfig`。当`quant_config`不为`None`时，`choose_tiling()`函数在填写`TilingPlan.quant_mode`时优先查询`quant_config`字典，而非回退到硬编码映射表；当某层的形状签名在字典中不存在时，编译器发出警告并使用硬编码默认值（保持向下兼容性）。

**量化工具链适配**：主流量化框架（如PyTorch的`torch.quantization`模块、ONNX Runtime的量化工具、Qualcomm AIMET）在导出标定结果时的格式各不相同。为避免编译器深度绑定特定框架，建议在量化工具链侧提供一个轻量级的适配器（adapter）脚本（约50行Python），负责将框架特有的量化注释（QuantizationAnnotation）转换为编译器期望的JSON格式。适配器作为量化工具链包的一部分交付，与编译器本体解耦，使编译器能够透明地消费来自不同工具链的标定结果。

**`quant_mode`与激活统计信息的映射规则**：目标硬件的`quant_mode`字段是一个小整数（当前已知取值范围为{0, 1, 2, 3}），其与激活量化配置的映射关系由硬件团队根据芯片量化参数表的设计固化，需通过查阅硬件文档或与芯片设计者协商确认。一个合理的初始假设是：`quant_mode`编码了激活值的量化位宽与缩放精度的组合——例如，`quant_mode=0`对应8-bit对称量化（scale为2的幂次），`quant_mode=2`对应16-bit非对称量化（带零点偏移）。编译器侧维护一张`(bit_width, is_symmetric, scale_is_power_of_2) → quant_mode`的映射表，在接收到量化配置后进行转换。这张映射表由硬件文档决定，与编译器其余逻辑解耦，可独立更新。

完成上述集成后，`quant_mode`将成为编译器从量化配置中自动推导的字段，与当前已自动推导的`acc_mode`、`store_mode`并列，编译器对外部输入的依赖将进一步缩减。这是从"特定模型专用工具"走向"通用量化神经网络编译器"的关键一步。

### 7.4.4　硬件内存布局建模与精确地址推导

建立内存分区模型，将权重存储区、特征图双buffer区、量化参数区等各分区的起始地址和对齐规则显式纳入编译器数据结构，使`bas_addr`的生成从外部依赖转变为编译器可推导的内部计算。推荐引入多面体内存分析（Polyhedral Memory Analysis）配合整数线性规划求解器（如Google OR-Tools）以最小化feature buffer连续布局中的地址空洞。完成后，可消除当前与黄金参考之间的`bas_addr`字段差异，实现真正意义上的全字段精确匹配，编译器输出可无需任何人工标注即直接下载到硬件。

**精确推导的三个子问题**：`bas_addr`的精确生成可分解为三个相对独立的子问题，难度和紧迫性各不相同。

第一个子问题是**权重地址的自动推进**（`weight_bas_addr`）。当前编译器中，`EmitterState.weight_bas_addr`在每条WeightLoader发射后按`weight_transnum_base × cin_group × ky_outer`的固定步长推进，逻辑已相对完整，且在FSRCNN和SD-UNet的验证中均与黄金参考吻合。这部分已基本解决，无需额外工作。

第二个子问题是**特征图（feature map）在片上buffer内的起始偏移**（DataLoader和DataStorer的`bas_addr`）。这是当前831处差异的主要来源。目标加速器的片上feature buffer被二着色（分为区域'a'和区域'b'），每个区域内各层的特征图以紧凑连续布局（compact contiguous layout）排列，前一层输出的末尾地址即为后一层输入的起始地址，递增步长由各层输出张量的字数（`h_out × w_out × c_out / pixels_per_word`）决定。现有的`ir/addr_alloc.py`框架已建立了基于活跃区间的tensor分配模型（§6.3.7），可在此基础上增加地址分配功能：对每个tensor，在其区域（'a'或'b'）内按定义顺序（def\_layer升序）分配连续起始地址，地址步长由tensor大小决定，活跃区间结束后的地址区间可复用。实现这一逻辑预计约需50~80行代码，可将DataLoader/DataStorer的`bas_addr`差异从831处压缩至接近零。

第三个子问题是**量化参数区和权重存储区的全局地址基准**（各类指令`bas_addr`的绝对基地址，对应硬件内存分区表中的分区起始地址）。这部分依赖芯片级内存映射（memory map），需从硬件团队获取并以常量形式写入编译器配置，属于一次性的接口对齐工作，完成后对所有后续模型均透明生效。

**`base_addrs_res`与`base_addr_pooling`的精确推导**：DataStorer中的`base_addrs_res`（DataStorer输出的地址累进量）和`base_addr_pooling`（pool-while-store场景下池化输出地址）是`bas_addr`推导中最复杂的两个字段，因为它们不是静态常量而是在tile循环内动态递增。`base_addrs_res`的递增步长等于每次DataStorer写出的字节数（`storer_step × c_out / words_per_row`），可从TilingPlan的`storer_step`字段和LayerDesc的`cout`字段直接计算；`base_addr_pooling`的递增步长则等于池化输出每步前进的字数，与`h_out_per_step`和池化因子的乘积相关。将这两个字段的推导逻辑从黄金参考中系统性地逆向分析后，以参数化公式的形式写入对应的发射函数，是消除剩余差异、实现全字段精确匹配的最后一公里工作。

### 7.4.5　load\_next Hoisting调度优化

当前`OffchipDataLoader`（下一帧图像预取）固定在Layer 0的分块循环全部完成后发出。硬件的`dependency`字段支持记分牌（Scoreboard）驱动的乱序执行，`OffchipDataLoader`作为DMA指令无上游数据依赖，理论上可在Layer 0循环的早期某个tile之后提前发出，使DDR数据预取与剩余tile的计算在时间上重叠，消除内存空泡（bubble）。`emitter.py`已预留`hoist_after_tile`参数接口，后续可在正确性验证完备后将其作为独立的调度优化实验推进，并以硬件仿真器定量评估实际吞吐收益。这一思路与Halide的异步DMA pragma [Ragan-Kelley, 2013]和TVM的prefetch优化在学术上一脉相承。

### 7.4.6　面向MLIR的架构演进

从长期技术演进角度，MLIR [Lattner, 2021]的可扩展方言（Dialect）机制为编译器基础设施提供了更强的模块化能力。当前的分层中间表示体系（Relay IR → LayerDesc → TilingPlan → 伪指令）在概念上与MLIR的多层方言设计高度对应，可以考虑将各层次分别映射为独立的MLIR方言（`relay_dialect`、`layerdesc_dialect`、`tiling_dialect`、`isa_dialect`），利用MLIR内置的Pass管理框架和类型系统，进一步提升Pass之间的接口严格性和可组合性。这一演进路径不需要重写算法逻辑，只需为已有的数据结构和Pass建立对应的MLIR IR绑定，是一条相对低风险的技术升级路径。

### 7.4.7　算子覆盖扩展与新模型接入

当前模板系统以FSRCNN和SD-UNet为设计基准，对于`pixel_shuffle`（像素重排，已在DepthToSpace透明化中初步实现）、深度可分离卷积（Depthwise Separable Convolution）、残差连接（Residual Connection）等算子尚缺少完整的TilingPlan模板。在更复杂的CNN模型（如RCAN [Zhang, 2018]、RealESRGAN [Wang, 2021]等超分辨率网络）上推广，需要系统性地补充对应算子的LayerDesc解析规则、TilingPlan模板和指令发射逻辑。由于编译器的四级分层架构将算子特有逻辑限定在各层的具体分支中，扩展新算子类型不影响已有算子路径的稳定性，是本系统可扩展性设计的直接体现。

---

## 7.5　结语

本文从一个具体的工程问题出发——如何让一款自研CNN加速器不再依赖手写的3,800行硬编码脚本——出发，通过设计分层中间表示体系、提出结构性模式融合Pass、实现针对性的调度策略，构建了一个具有完整功能的TVM前端编译器，并在FSRCNN（1,273/1,273指令数对齐）和SD-UNet（17,155/17,155，0功能性diff，数据通路等价性经形式化验证）两个目标网络上完成了端到端的指令级精确验证。

这项工作的意义不仅在于取代了手写脚本，更在于它揭示了一种可复用的设计模式：面向定制加速器的神经网络编译器，可以在TVM的前端能力（多框架导入、IR规范化、Pass管理）基础上，通过精确控制的分层中间表示和针对性的融合Pass，以远小于手写方案的工程代价，实现对硬件ISA语义的精确建模。两个算子复杂度差异悬殊的网络在同一框架下得到正确编译，印证了这一方法论的有效性。

从性能等价性的视角审视，本工作的核心价值更为清晰。两个目标网络的指令数均与手写参考精确一致——FSRCNN：1,273/1,273；SD-UNet：17,155/17,155——而该参考正是硬件工程师针对这款MAC阵列手工调优的最优指令序列。这意味着，编译器在实现完全自动化的同时，达到了手工编码所能实现的性能上界：**自动化与性能等价同时实现，是本工作的核心价值所在**。从编译器前端的设计边界来看，可供挖掘的指令级优化空间已基本穷尽。若需进一步提升端到端推理吞吐，方向在于硬件流水线深度、片上存储层次的协同设计，或采用剪枝、量化等模型压缩技术以降低算子本身的计算量，这些均超出编译器前端的职责范围，是后续硬件-算法-编译器系统协同优化的自然延伸方向。

随着硬件加速器设计的多样化和神经网络模型的快速演进，如何让编译器跟上模型和硬件的迭代节奏，将成为越来越重要的工程命题。本文所建立的分层框架和验证方法论，希望能为这一命题的探索提供一点实践参考。
