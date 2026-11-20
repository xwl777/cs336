# CS336 · 从分词到语言模型训练

这份仓库记录我学 CS336 时写的代码、做过的实验和课程笔记。截至 2026 年 11 月 20 日，主要实现集中在作业 1、2、4。作业 3 有 Scaling Law 学习笔记，作业 5 已整理课程框架和策略梯度笔记。

## 作业进度

### 1 · Basics — 从 BPE 到文本生成

- **已实现**　字节级 BPE 训练与编解码、带 RoPE 的因果 Transformer（RMSNorm、SwiGLU）、AdamW、梯度裁剪、余弦学习率、训练循环和 top-p 采样。保留了 TinyStories 的词表与合并规则，以及 TinyStories、OpenWebText 的训练配置。[模型与训练代码](assignment1-basics/cs336_basics/) · [运行脚本](assignment1-basics/scripts/)
- **待补**　模型、数值工具、优化器和序列化四组测试合计 16 项通过、3 项失败。三处分别是无 RoPE 注意力的权重名不匹配、独立 SiLU 测试接口尚未接入、checkpoint 加载返回值与测试约定不一致。全套测试在检查无效设备 `cuda:99` 的取批用例时进程崩溃，仍需完整复测。仓库中也缺少各训练配置对应的验证损失和最终对比。[测试](assignment1-basics/tests/) · [BPE 实现笔记](notes/Assignment1.md)

### 2 · Systems — Attention、通信与显存

- **已实现**　FlashAttention 的 PyTorch 和 Triton 前反向，逐参数同步、梯度展平和通信重叠三种 DDP，以及分片优化器和 FSDP。已有双进程小模型的 CPU 实验结果，FSDP 指南记录了 5 个 CPU/Gloo 用例连续五轮通过。[系统实现](assignment2-systems/cs336_systems/) · [分片优化器实验](assignment2-systems/benchmark/SHARDED_OPTIMIZER_PROFILE.md) · [FSDP 验证记录](assignment2-systems/benchmark/FSDP_GUIDE.md)
- **待验证**　本次复测中 PyTorch 前向通过，反向用例因本机 OpenMP 运行库重复加载而中断。Triton 正确性、FlashAttention 延迟，以及双 GPU 上 XL 模型的耗时和峰值显存仍待实测。目前 XL 报告给出的是模型状态存储量估算。[FlashAttention 测量方案](assignment2-systems/benchmark/FLASH_BENCHMARK.md)

### 3 · Scaling Laws — 已整理理论笔记

笔记整理了数据量与 loss 的关系、重复数据、临界 batch size，以及 Kaplan 和 Chinchilla 的计算预算分配。仓库中尚无作业 3 的实验代码、拟合曲线或报告。[Scaling Law 笔记](notes/7.%20Scaling%20law.md)

### 4 · Data — 从 Common Crawl 到训练文本

- **已实现并运行**　HTML 正文抽取、语言识别、个人信息脱敏、有害内容与质量过滤，以及精确行去重和 MinHash 近似去重。WET 基线流水线另用全文哈希去重，在一份真实 WET 文件中读入 **19,637 篇**，最终保留 **1,659 篇**。串行与双进程输出解压后逐字节一致。[实现](assignment4-data/cs336_data/) · [过滤实验记录](assignment4-data/WET_FILTERING_GUIDE.md) · [串并行对照](assignment4-data/data/filtered_wet_baseline/serial_parallel_check.json)
- **待补**　本地完整测试 48 项通过，7 项失败均来自尚未实现的 Gopher 质量过滤接口。下一阶段还要把过滤语料分词，用固定配置训练语言模型，并用 Paloma 验证集的困惑度比较数据版本。[质量过滤测试](assignment4-data/tests/test_quality.py)

### 5 · Alignment — 已整理框架和算法笔记

已有课程提供的 prompt 模板、奖励评分工具、安全评测脚本和测试，以及从策略梯度到 PPO、TRPO、DPO、GRPO 的推导笔记。作业的 12 个测试接口仍为占位实现，待补的主要内容包括 SFT 数据打包、response log-prob、组内奖励归一化、策略梯度与 GRPO 训练步、DPO loss 和 MMLU/GSM8K 答案解析。[待实现接口](assignment5-alignment/tests/adapters.py) · [策略梯度笔记](policy_gradient_ppo_trpo_dpo_grpo.md)

## 课程笔记

- **分词与模型**　[Tokenization](notes/1.Tokenization.md) · [PyTorch 与 einops](notes/2.Pytorch（einops）.md) · [Transformer 架构](notes/3.Architecture.md) · [Attention 变体](notes/4.Attention%20alternative.md) · [作业 1 的 BPE 优化记录](notes/Assignment1.md)
- **训练系统**　[GPU、Kernel 与 Triton](notes/5.GPU,Kernel,Triton.md) · [并行训练](notes/6.Parallelism.md) · [Scaling Law](notes/7.%20Scaling%20law.md)
- **推理、评测与数据**　[Inference](notes/8.%20Inference.md) · [Evaluation](notes/9.%20Evaluation.md) · [Data](notes/10.%20Data.md)

## 复现实验

每份作业有独立的依赖和运行说明。重新运行实验时，需要按对应作业的说明准备原始数据和模型权重。仓库保留了代码、配置、笔记、测试样例和实验摘要。作业 4 的上述测试使用了本地 fastText 权重，作业 5 的两个 tiny-GPT2 测试权重也需要单独准备。
