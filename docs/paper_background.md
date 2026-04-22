# 背景与相关工作

## 第一部分：卷积神经网络与图像超分辨率

深度学习的崛起深刻改变了计算机视觉领域的面貌。卷积神经网络（Convolutional Neural Network，CNN）作为这一变革的核心技术，自20世纪80年代末由LeCun等人提出雏形以来[1]，历经数十年演进，已成为图像识别、目标检测、图像生成等任务的基础构件。LeNet的提出奠定了"卷积层-池化层-全连接层"的基本范式；2012年，AlexNet借助GPU算力在ImageNet竞赛上以压倒性优势夺冠，开启了深度学习的工程化时代[2]。此后，VGGNet通过加深网络层数探索了深度与精度的关系，而残差网络（ResNet）[3]则以跳跃连接（skip connection）解决了深层网络的梯度消失问题，将网络深度推至150层以上，性能持续突破。DenseNet[4]进一步将密集连接引入网络设计，使每一层都能直接获取前序所有层的特征，在参数效率与精度之间取得了良好平衡。

图像超分辨率（Super Resolution，SR）是CNN应用中一个极具代表性的低层视觉任务，其目标是从低分辨率（Low Resolution，LR）输入重建高分辨率（High Resolution，HR）图像。Dong等人于2014年提出的SRCNN[5]是将深度学习引入超分辨率领域的开创性工作，其端到端的训练范式一举超越了传统稀疏编码方法。为克服SRCNN推理速度慢的局限，同一团队于2016年提出FSRCNN（Fast Super-Resolution Convolutional Neural Network）[6]，通过在低分辨率空间完成特征提取与非线性映射、在最后一层以亚像素卷积（pixelshuffle）实现上采样，将推理速度提升数十倍，同时保持可比的重建质量。FSRCNN的网络结构可分为四个功能模块：特征提取层（Feature Extraction，3×3卷积）、压缩层（Shrinking，1×1卷积降维）、非线性映射层（Mapping，若干3×3卷积）以及上采样层（Deconvolution，基于PixelShuffle实现）。这种分阶段的"先压缩、后映射、再上采样"设计不仅降低了计算复杂度，也为硬件加速提供了清晰的算子边界。

在后续发展中，感知质量驱动的超分辨率网络不断涌现。ESRGAN[7]引入生成对抗网络（GAN）框架，显著提升了重建图像的感知真实感。与此同时，可变形卷积（Deformable Convolution）[8]的提出为超分辨率网络带来了新的建模能力。Dai等人于2017年指出，标准卷积核的固定采样网格限制了网络对几何形变的建模能力；他们提出在每个采样位置叠加一个可学习的偏移量（offset），使卷积核能够自适应地关注图像中语义相关的区域。DCNv2[9]进一步引入可调制因子（modulation），增强了偏移量的表达能力。可变形卷积在超分辨率任务中表现出色：通过对低分辨率特征图进行自适应采样，网络能够更精确地捕捉边缘、纹理等高频细节，显著提升重建精度。

本文所针对的FSRCNN变体（`models_new_930.py`）在经典FSRCNN结构的基础上，将非线性映射层中的标准卷积替换为可变形卷积（DeformableConv2d），并设计了专用的偏移量生成子网络（OffsetGenerator）：该子网络先以AvgPool2d对特征图下采样，再经一个3×3卷积预测偏移场，最后通过`repeat_interleave`插值上采样至原始分辨率，并做边界复制填充（replicate padding）。每个DeformableConv2d模块内部，偏移量生成与主卷积分支并行运行，最终以双线性插值实现可变形采样。这一设计使FSRCNN在轻量级模型规模下兼具几何形变建模能力，但也对编译器前端提出了新的挑战：可变形卷积在计算图层面由多个基础算子组成，如何将其整体识别并映射到硬件加速路径，是本文的核心问题之一。

## 第二部分：专用深度学习硬件加速器

深度学习推理在数据中心和边缘端的大规模部署，催生了对专用硬件加速器的迫切需求。相比通用处理器（CPU），GPU在大批量矩阵运算上具有显著优势，但其高功耗与内存带宽约束使之难以满足资源受限场景的部署要求。专用集成电路（Application-Specific Integrated Circuit，ASIC）形式的深度学习加速器，通过面向特定算子（尤其是卷积）定制化设计数据通路与存储层次，能够在远低于GPU功耗的条件下实现接近或超越GPU的吞吐量与能效[10]。

在学术界与工业界涌现的主流加速器设计中，Google的TPU（Tensor Processing Unit）[11]是最具代表性的工业级脉动阵列（Systolic Array）实现。脉动阵列以规则的二维处理元件（Processing Element，PE）阵列为核心，数据在相邻PE之间以流水方式传播，每个PE完成一次乘加运算（Multiply-Accumulate，MAC）。脉动阵列的优势在于极高的运算密度与良好的数据复用性：权重固定（weight stationary）或输出固定（output stationary）等不同数据流策略，可针对不同算子特性最大化片上数据复用，降低对外部存储带宽的依赖。MIT的Eyeriss[12]系统性地提出了"行固定"（row stationary）数据流策略，通过对卷积运算中多种形式数据复用的联合优化，在能效方面达到了当时的最优水平，并引发了学术界对数据流设计空间的广泛探索[13]。寒武纪MLU系列和华为昇腾达芬奇架构则代表了国内芯片厂商在AI加速器领域的重要突破，分别针对推理与训练场景进行了深度优化。

本文所面向的自研CNN加速器采用MAC阵列为核心计算单元，配合多级片上存储层次：片外DDR负责模型权重与中间特征图的持久化存储；片上分为offchip_input_buffer、weight_buffer、quant_buffer等一级缓冲，以及直接服务MAC阵列的line_buffer；MAC阵列运算结果存储在acc_reg中，经DataStorer后处理后回写input buffer（a/b路双缓冲）。这种层次化存储设计的核心动机在于：DDR访问延迟远高于片上SRAM，且带宽有限。通过将一个tile的输入特征、权重、量化参数一次性加载至片上，在片上完成全部MAC运算后再将结果写回，可以将绝大多数内存访问转化为高效的片上操作，极大缓解内存带宽瓶颈。这一内存层次结构直接决定了编译器的tiling策略：tile的尺寸必须与各级buffer的物理容量严格对齐，且行方向（H）、列方向（W）、输入通道（Cin）与输出通道（Cout）的分块粒度（如4行、32/64/128列、8/16/32通道）均由硬件约束固定，编译器前端需要理解并遵循这些约束以生成正确的指令序列。

此外，该加速器原生支持可变形卷积的硬件加速：OffsetLoader指令将偏移量加载至专用的offset_reg，WeightLoader通过`is_bilinear_bicubic`字段选择双线性插值模式，使MAC阵列能够直接完成可变形采样与卷积的联合计算。这一硬件特性的正确利用，高度依赖编译器前端能够准确识别并保留可变形卷积的完整计算语义。

## 第三部分：AI 编译器

深度学习框架的快速多元化（PyTorch、TensorFlow、ONNX、PaddlePaddle等）与硬件加速器的爆发式增长，共同构成了"N×M问题"：若要为每一对（框架，硬件）单独实现优化编译路径，工程成本将以乘积量级增长，难以为继。AI编译器（AI Compiler）的出现正是为了在这一多样性矩阵中引入统一的抽象层，以中间表示（Intermediate Representation，IR）为枢纽，将前端的框架导入与后端的硬件代码生成解耦，从而以N+M的线性成本应对N×M的适配挑战[14]。

在主流AI编译器中，TVM[15]是当前学术界引用最广泛、工程实践最成熟的开源系统。TVM的架构可分为三个层次：**前端**（Frontend）负责从各主流框架（PyTorch、TensorFlow、ONNX、MXNet等）导入计算图，并统一转换为Relay IR；**中端**（Middle-end）以Relay或Relax IR为载体，运行一系列图级优化Pass，包括算子融合（operator fusion）、常量折叠（constant folding）、死代码消除（dead code elimination）、形状推断（shape inference）等；**后端**（Backend）通过Schedule和TIR（Tensor IR）将优化后的计算图降级为目标硬件的可执行代码，支持CUDA、Metal、Vulkan、LLVM，以及自定义硬件的C/汇编输出。Relay IR[16]采用函数式计算图表示，每个算子对应一个纯函数节点，类型系统保证了静态形状推断的可靠性；Relax IR则进一步引入了动态形状支持与更灵活的控制流表达，代表了TVM下一代IR的发展方向。

在硬件无关的图优化层面，Halide[17]最早系统性地提出了"算法-调度"分离原则：计算语义（what to compute）与执行策略（how to compute）在语言层面明确解耦，使工程师可以在不改动算法的前提下，通过调整调度策略（tile、vectorize、unroll、parallel等）探索大规模优化空间。这一思想深刻影响了后续AI编译器的设计。MLIR[18]由LLVM社区推动，以"方言"（dialect）机制提供了一套可扩展的多层IR框架，允许不同抽象层次的优化共存于同一系统中，已成为众多工业级AI编译器的基础设施。XLA（Accelerated Linear Algebra）作为TensorFlow的官方编译器，在Google内部以JIT和AOT两种模式支持TPU、GPU等后端，在大规模分布式训练中被广泛采用。

Tiling（分块）优化是AI编译器中最关键的循环变换策略之一。卷积运算的计算量与输入、权重的尺寸呈多项式增长，若将整个特征图一次性加载至片上，往往远超片上SRAM容量。通过将H、W、Cin、Cout维度分别切分为固定大小的tile，每次仅加载一个tile所需的数据，可以将DDR访问次数降至最低，同时最大化数据在片上的复用次数，从而有效利用内存带宽、降低访存能耗[13]。Ansor[19]通过层次化搜索空间与进化算法自动化了tiling参数的搜索过程，在多种硬件上达到了接近手工调优的性能水平。然而，对于本文所针对的自研加速器，tiling的粒度（如4行、32列、8通道）由硬件物理约束严格决定，不存在连续搜索空间，因此需要编译器前端以确定性方式推断并应用这些约束，而非依赖自动调优。

当前AI编译器面临的核心挑战集中在以下三点：其一，**自定义算子支持**——可变形卷积、注意力机制等新型算子难以用通用基础算子高效表达，需要编译器提供自定义算子注册与识别机制；其二，**专用硬件精确映射**——不同加速器的指令语义、数据布局约束和内存层次差异显著，通用代码生成路径往往无法生成语义正确、性能最优的指令序列；其三，**图-指令语义鸿沟**——从高层计算图到底层微指令序列，中间横跨多个抽象层次，每一层的信息丢失都可能导致最终输出偏离硬件预期。本文所设计的TVM编译器前端，正是在上述挑战的背景下，针对FSRCNN超分辨率网络与自研CNN加速器，提出了一套完整的前端设计方案，涵盖多框架模型导入、可变形卷积一等公民算子识别、面向硬件约束的图级优化，以及与后端tiling/codegen接口的规范化对接。

---

## 参考文献

[1] LeCun Y, Boser B, Denker J S, et al. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1989, 1(4): 541-551.

[2] Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks. In Proceedings of NeurIPS, 2012: 1097-1105.

[3] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. In Proceedings of CVPR, 2016: 770-778.

[4] Huang G, Liu Z, van der Maaten L, et al. Densely connected convolutional networks. In Proceedings of CVPR, 2017: 4700-4708.

[5] Dong C, Loy C C, He K, et al. Learning a deep convolutional network for image super-resolution. In Proceedings of ECCV, 2014: 184-199.

[6] Dong C, Loy C C, Tang X. Accelerating the super-resolution convolutional neural network. In Proceedings of ECCV, 2016: 391-407.

[7] Wang X, Yu K, Wu S, et al. ESRGAN: Enhanced super-resolution generative adversarial networks. In Proceedings of ECCV Workshops, 2018.

[8] Dai J, Qi H, Xiong Y, et al. Deformable convolutional networks. In Proceedings of ICCV, 2017: 764-773.

[9] Zhu X, Hu H, Lin S, et al. Deformable ConvNets V2: More deformable, better results. In Proceedings of CVPR, 2019: 9308-9316.

[10] Sze V, Chen Y H, Yang T J, et al. Efficient processing of deep neural networks: A tutorial and survey. Proceedings of the IEEE, 2017, 105(12): 2295-2329.

[11] Jouppi N P, Young C, Patil N, et al. In-datacenter performance analysis of a tensor processing unit. In Proceedings of ISCA, 2017: 1-12.

[12] Chen Y H, Emer J, Sze V. Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks. In Proceedings of ISCA, 2016: 367-379.

[13] Parashar A, Raina P, Shao Y S, et al. Timeloop: A systematic approach to DNN accelerator evaluation. In Proceedings of ISPASS, 2019: 304-315.

[14] Li M, Liu Y, Liu X, et al. The deep learning compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed Systems, 2021, 32(3): 708-727.

[15] Chen T, Moreau T, Jiang Z, et al. TVM: An automated end-to-end optimizing compiler for deep learning. In Proceedings of OSDI, 2018: 578-594.

[16] Roesch J, Lyubomirsky S, Weber L, et al. Relay: A new IR for machine learning frameworks. In Proceedings of MAPL Workshop at PLDI, 2018.

[17] Ragan-Kelley J, Barnes C, Adams A, et al. Halide: A language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. In Proceedings of PLDI, 2013: 519-530.

[18] Lattner C, Amini M, Bondhugula U, et al. MLIR: Scaling compiler infrastructure for domain specific computation. In Proceedings of CGO, 2021: 2-14.

[19] Zheng L, Jain A, Wong E, et al. Ansor: Generating high-performance tensor programs for deep learning. In Proceedings of OSDI, 2020: 863-879.
