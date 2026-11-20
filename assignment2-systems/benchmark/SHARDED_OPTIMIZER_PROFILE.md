# 优化器状态分片：显存与训练速度对比

脚本：[benchmark_sharded_optimizer.py](../scripts/benchmark_sharded_optimizer.py)。
实现：[sharded_optimizer.py](../cs336_systems/sharded_optimizer.py)。

## 当前完成到哪里

已完成双模式 profiling 脚本、真实 xl 模型形状的存储量计算和两进程小模型 CPU 冒烟测试。当前机器的 PyTorch 报告 `cuda.is_available() == False`、CUDA 设备数为 0，因此没有双 GPU xl 的显存峰值或每步耗时实测；下面的显存数字是模型状态存储量估算，时间示例是明确假设下的估计，不能当作实测提交。

## 如何在双卡机器上运行

在 `assignment2-systems` 目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/benchmark_sharded_optimizer.py --mix_precision
```

默认配置沿用现有 DDP benchmark：24 层、d_model=1600、25 个头、d_ff=6400、词表 10000、序列长度 1024；每卡 micro-batch=4、梯度累积 8 次，每个更新处理 65536 tokens。参数、梯度和 AdamW 的两个矩保持 FP32；`--mix_precision` 只启用 BF16 autocast，没有独立 FP32 master 参数副本。

两种模式使用相同 `OverlapDDP`、相同数据种子和 `cs336_basics.optimizer.AdamW`，每种模式重新启动独立进程。也可以不带 `--mix_precision` 比较纯 FP32，两种模式必须保持相同设置。

输出目录默认是 `benchmark/sharded_optimizer_xl/`：

- `unsharded.json`、`sharded.json`：每个 rank 的三处内存快照、首步/稳定步、逐步耗时。
- `comparison.json`：完整两组报告、平均步耗时之比和变慢百分比。
- `estimate.json`：只在 `--estimate_only` 模式生成，记录解析存储量，明确不是真实峰值。

无需 GPU 的存储计算：

```bash
uv run python scripts/benchmark_sharded_optimizer.py --estimate_only
```

## 测量口径：不要把“现在占多少”混成“之前最高多少”

每个 CUDA 快照先同步设备，再记录：

- `allocated_bytes`：快照时 PyTorch 活跃分配。
- `reserved_bytes`：allocator 保留显存，包含缓存，不等于活跃 tensor 内存。
- `phase_peak_allocated_bytes`：当前阶段的活跃分配峰值。初始化阶段、forward/backward 阶段、optimizer 阶段分别重置峰值计数。
- `peak_since_initialization_bytes`：综合各阶段与 warmup 的历史最高活跃分配，不能随 tensor 释放而下降。
- `parameters_bytes`、`gradients_bytes`、`optimizer_states_bytes`、`buffers_bytes`：快照时各类真实 tensor storage 的字节数，对共享 storage 去重。
- `other_live_bytes`：活跃分配减去上述组成；不能把它全部称为“激活”，其中也可能有临时 tensor、allocator 对齐等。

初始化快照在模型、DDP wrapper 和优化器构造完成后采集，尚未 forward。step 前快照在最后一次 backward 和梯度同步之后采集，step 后快照在本地更新与参数广播之后采集。释放最后一份 logits/loss 引用，避免把人为保留的输出计入 optimizer 阶段。

**阶段峰值不能直接用快照组成相加来还原。** 例如 forward 的激活可能已经释放，但 forward/backward 阶段的最高占用仍被峰值计数记住；初始化也可能有临时分配。

PyTorch CUDA allocator 指标不包含所有 CUDA context 和 NCCL 内部分配，所以它与 `nvidia-smi` 的进程显存不必相等。CPU 冒烟输出中的 CUDA 内存字段为 null，只能用于验证流程与 tensor 存储组成。

## 为什么要同时记录首步与稳定步

AdamW 构造时状态为空；第一次 step 才创建 m、v。因此首步 step 前两种模式的参数/梯度组成相同，而首步 step 后才出现状态分片带来的节省。

稳定步 step 前，上一轮的 m、v 已经存在，分片的节省在 forward/backward 期间也存在，可能降低其峰值。不能只测首步 backward 就断言分片不省内存。

计时单独进行，不含内存快照与 storage 遍历；3 个 warmup 后测 10 步，每步包括 forward、backward、梯度同步、优化器更新及参数广播，并取各 rank 的最大时间后求均值/标准差。排除迭代开始处的 barrier，保证 slowest rank 的耗时代表完整分布式迭代。

## (a) 可供修改的 3 句预估回答

按代码的实际 xl 结构（1,015,118,400 个参数）和 FP32 状态计算，每卡初始化的参数存储约 3.78 GiB，首步 optimizer 前的参数加梯度约 7.56 GiB，首步 optimizer 后普通 AdamW 约 15.13 GiB，而分片后 rank 0/1 约 11.46/11.23 GiB；这些是常驻模型状态组成的预估，不是 GPU 峰值实测。两者都保留约 3.78 GiB 参数与 3.78 GiB 梯度，普通优化器另有约 7.56 GiB 的 m、v，分片后状态分别约 3.90/3.66 GiB，这符合“只减少优化器状态、不减少完整参数和梯度”的预期。实际初始化、backward 和 optimizer 阶段峰值还会包含激活、临时缓冲及其他分配；稳定步在 step 前就已有优化器状态，不能与首步的 7.56 GiB 混用。

这里“一个参数”是一个完整 Parameter tensor，按数量轮流分配；rank 0 拥有 523,558,400 个元素，rank 1 拥有 491,560,000 个元素，故不是精确对半。模型状态总量约减少 24.2%–25.8%，而不是全部训练显存减半。

## (b) 可供修改的 3 句预估回答

没有目标 GPU 型号和实际带宽，无法给出可信的双卡 xl 秒数；一个明确的估算情景是普通训练每步 4.00 秒、分片减少的本地优化器时间约 0.02 秒、额外参数广播有效聚合带宽为 50 GiB/s，则广播约需 3.78/50=0.076 秒，分片每步约 4.06 秒、变慢约 1.4%。这只是忽略小 collective 启动开销、网络竞争与 owner 不均衡的条件估算，而不是硬件预测或测量结果。我们的实现完整 all-reduce 梯度之后又逐参数广播，因此在通信开销超过本地更新节省时预计会变慢；准确结果应使用脚本输出的 `mean_step_ms` 及 `sharded_to_unsharded_time_ratio`。

50 GiB/s 是整轮广播的有效聚合传输带宽假设，不是某张卡的单向峰值带宽。一般关系是：`T_sharded ≈ T_normal − 节省的本地更新时间 + 广播时间 + 启动与等待开销`；内存省下来也可能允许更大的 batch，但本实验固定 batch 以保证公平比较。

最终两进程 CPU 小模型（6,224 个参数）冒烟实测：普通 11.25 ms/步，分片 14.30 ms/步；仅用于验证脚本流程，不能外推成 GPU xl 的训练时间。完整记录见 `sharded_optimizer_cpu_smoke/comparison.json`。

## (c) 可用的 3 句回答

我们的实现与 ZeRO Stage 1 都分片优化器状态并保留完整模型，但我们按整个参数 tensor 轮流分配，可能使各 rank 状态大小不均衡；论文按状态分区讨论更均衡的 1/N 分配，且其混合精度内存公式包含 FP32 master 参数，而我们的 FP32 参数本身就是更新对象，没有另存 master 副本。通信上，我们先对完整梯度做 all-reduce，再逐参数广播更新结果，多出了一轮全模型参数同步；按论文的大规模归一化口径约为 3Ψ，而论文的 ZeRO P_os 目标为约 2Ψ，并通过协调梯度归约和更新后参数收集避免多余的梯度分发。因而本实现的状态分片思想相同，但没有实现论文级别的通信调度和打包，不能直接套用其“通信量与普通 DP 相同”的结论。[ZeRO 原文 §§5.1、5.2、7](https://arxiv.org/html/1910.02054v3)

两卡且参数/梯度同为 FP32 时，用全节点发送量计数（不再把接收量重复相加）：标准梯度 all-reduce 约发送 2P，额外广播发送 P，本实现合计 3P，约是仅梯度 all-reduce 的 1.5 倍，其中 P 是完整参数字节数。真实 NCCL 算法、拓扑和小 tensor 启动开销会影响耗时，通信量增加 50% 不等于整步慢 50%。

参考资料：论文的 §5.1 描述优化器分区与更新后 all-gather，§5.2/§7 展开将梯度归约至负责分区、再收集更新参数的通信安排；不能把 Stage 1 和 Stage 2 的“释放非本地梯度存储”混为一谈。[ZeRO 论文](https://arxiv.org/html/1910.02054v3)
