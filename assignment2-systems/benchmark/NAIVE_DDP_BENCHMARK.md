# Naive DDP Benchmark 代码详解

对应脚本：[`scripts/benchmark_naive_ddp.py`](../scripts/benchmark_naive_ddp.py)
被测对象：[`cs336_systems/naive_ddp.py`](../cs336_systems/naive_ddp.py) 中的 `NaiveDDP`、
[`cs336_systems/flat_ddp.py`](../cs336_systems/flat_ddp.py) 中的 `FlatDDP`

---

## 1. 整体思路

这个脚本回答作业里的问题：**用 naive DDP 训练 xl 模型时，一个训练 step 花多长时间？其中多少时间花在梯度通信上？**

做法：

1. 启动 2 个进程（1 节点 × 2 GPU），每个进程各建一份完整的 xl 模型；
2. 用 `NaiveDDP` 包起来，跑若干个训练 step；
3. 每个 step 里用秒表分别计时「计算部分」和「梯度同步部分」；
4. 最后汇总输出，并写 `summary.json` / `results.csv`。

运行方式（在 GPU 机器上）：

```bash
cd assignment2-systems
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/benchmark_naive_ddp.py --ddp_impl naive --mix_precision
CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/benchmark_naive_ddp.py --ddp_impl flat  --mix_precision
```

---

## 1.5 FlatDDP：单次批量 all-reduce（minimal DDP with flat gradients）

作业下一问要求把「逐参数 all-reduce」改成「拼接成一个扁平大 tensor、只发 1 次 all-reduce」，
实现见 `cs336_systems/flat_ddp.py`：

```python
grads = [p.grad for p in parameters]                     # 300+ 个梯度张量
flat  = torch._utils._flatten_dense_tensors(grads)       # 拷进一个连续大 buffer（xl: ~3.1 GB fp32）
dist.all_reduce(flat)                                    # 唯一一次集合调用
flat.div_(world_size)
for p, view in zip(parameters, _unflatten_dense_tensors(flat, parameters)):
    p.grad = view                                        # unflatten 返回 flat 的 view，零拷贝写回
```

**为什么更快**：每次 all-reduce 调用有固定开销（kernel 启动、NCCL 协议握手、跨 rank 同步），
naive 版本对 xl 要发 300+ 次调用，其中大量是小消息（RMSNorm 权重仅 1600 元素 = 6.4 KB），
延迟主导、带宽用不满。flat 版本总字节数不变，但只有 1 次调用、且是大消息，
ring 算法可以流水线打满 interconnect 带宽。

**代价**：flatten/unflatten 各引入一次全量拷贝（写入 flat buffer）；需要额外一块连续显存（约等于梯度本身大小）；仍然与 backward 零重叠（后续 bucket + overlap 的改进方向）。

**正确性验证**：CPU/gloo 双进程对比实验中，同一 rank 上 naive 与 flat 同步后的梯度逐位一致（max-diff = 0）。

benchmark 脚本用同一个 `--ddp_impl {naive, flat, none}` 开关切换三种模式，计时方法学完全一致，
结果分别写入 `benchmark/<impl>_ddp_xl/`，直接对比两组的
`mean_grad_sync_ms` / `grad_sync_fraction` 即可回答 deliverable。

本机 CPU 冒烟测试（toy 模型）参考：naive 通信 5.3 ms（step 占比 ~55%）vs flat 0.5 ms（~10%），
300+ 次小集合调用合成 1 次大调用的效果方向正确；真实 xl + 2 GPU 数字待在 GPU 机器上测。

---

## 2. 逐段解释

### 2.1 模块 docstring（第 1–19 行）

说明脚本用途、xl 模型的具体超参、运行命令，以及全局 batch 的计算公式：

```
全局 batch token 数 = world_size × micro_batch_size × grad_accum_steps × context_length
```

因为要测的是**时间**而不是训练效果，所以数据用随机 token，不需要真实数据集。

### 2.2 导入（第 21–39 行）

| 导入 | 作用 |
|---|---|
| `torch.distributed`（`dist`） | 进程间通信：all_reduce、barrier、broadcast、all_gather |
| `torch.multiprocessing`（`mp`） | `mp.spawn` 启动多进程，每个进程绑一张 GPU |
| `cs336_basics.*` | assignment1 里自己实现的 `Transformer`、`AdamW`、`cross_entropy` |
| `cs336_systems.naive_ddp` | 被测的 `NaiveDDP`（逐参数 all-reduce） |

### 2.3 `XL_PRESET`（第 41–49 行）

Section 2.1.2 里 xl 模型的标准配置（约 774M 参数）：

```python
d_model=1600, num_heads=25, num_layers=24, d_ff=6400,
context_length=1024, vocab_size=10000, rope_theta=10000.0
```

这些值后面会变成命令行参数的默认值，所以也可以用 `--d_model` 等参数改成别的规模做对比。

### 2.4 建模和造数据（第 52–69 行）

- `build_model`：按参数在指定设备上构造 fp32 的 `Transformer`。
- `make_batch`：随机生成输入 `x` 和目标 `y`（形状 `[micro_batch_size, context_length]`），**直接生成在 GPU 上**，避免数据加载成为计时瓶颈。

### 2.5 `synchronize`（第 72–74 行）—— 计时方案的关键

CUDA 操作是**异步**的：Python 调用 `loss.backward()` 会立刻返回，GPU 其实还在算。如果不用 `torch.cuda.synchronize()` 把 CPU 阻塞到 GPU 真正干完活，秒表测出来的时间就是假的。

CPU 模式下这个函数什么都不做（所以脚本在 Mac 上也能跑通做冒烟测试）。

### 2.6 `train_step`（第 77–111 行）—— 一个完整的训练 step

时间线（每一段前后都有 `synchronize` + `perf_counter` 打点）：

```
zero_grad
   │
   ├─ t_compute_start ──────────────────────────────┐
   │  for _ in range(grad_accum_steps):             │
   │      造 batch → forward → loss → backward      │ 计算阶段
   │                                                │
   ├─ t_sync_start ─────────────┐                   │
   │  finish_gradient_sync()    │ 通信阶段           │
   ├─ t_sync_end ───────────────┘                   │
   │  optimizer.step()                              │
   ├─ t_compute_end ────────────────────────────────┘
```

要点：

- **第 83 行** `zero_grad(set_to_none=True)`：在计时开始之前做，不算进 step 时间。
- **第 88–93 行** 梯度累积循环：做 `grad_accum_steps`（默认 8）次微批量的 forward + backward。loss 除以累积步数，累积出的梯度等价于对一个大 batch 求平均。用累积是因为 xl 模型 fp32 下一次性放 0.5M token 的 batch 会爆显存。
- **第 98–99 行** `finish_gradient_synchronization()`：就是 `naive_ddp.py` 里的实现——对**每一个参数梯度逐个**做同步 `all_reduce(SUM)` 再除以 world_size。naive 就 naive 在：每个参数一次通信、完全串行、和 backward 零重叠。
- **第 109–110 行** 返回值：
  - `sync_seconds = t_sync_end − t_sync_start`：纯梯度通信时间；
  - `compute_seconds = 总时长 − 通信时长`：forward + backward + optimizer。

### 2.7 `worker`（第 114–239 行）—— 每个进程执行的主函数

#### 初始化（第 115–140 行）

- `rank=0` 绑 GPU 0，`rank=1` 绑 GPU 1（`torch.cuda.set_device`）。
- `init_process_group`：GPU 用 **NCCL** 后端，CPU 调试用 **gloo**。之后各 rank 才能互相通信。
- 所有 rank 用同一个随机种子 → 初始权重相同（`NaiveDDP.__init__` 里还会从 rank 0 广播一遍参数做双保险）。
- 只有「多进程且 `--ddp_impl != none`」时才创建 DDP 包装（`naive` → `NaiveDDP`，`flat` → `FlatDDP`）；`none` 模式下 `ddp_model=None`，`train_step` 里跳过同步——这是**纯计算对照组**，可验证「DDP 开销 = 有同步 − 无同步」。
- **先包装、后建 AdamW**（`NaiveDDP` docstring 要求的顺序）。
- `--mix_precision` 时 forward/loss 用 bf16 autocast（作业里 xl 就是这么训的）；否则用 `nullcontext()` 占位。

#### 统计信息（第 142–143 行）

数参数量和梯度字节数。xl 模型 fp32 梯度约 **774M × 4B ≈ 3.1 GB**——这就是每个 step 要 all-reduce 的数据量，可以和 `benchmark_all_reduce.py` 测的带宽互相印证。

#### 预热（第 145–148 行）

先跑 3 个不计时的 step。第一个 step 包含 CUDA kernel 编译/缓存、NCCL 建立通信链路等一次性开销，不预热会严重污染测量结果。

#### 正式测量（第 150–160 行）

- 每个 step 前先 `dist.barrier()`：让两个进程对齐到同一起跑线，避免一快一慢导致 all_reduce 的**等待时间**被误算进「通信时间」。
- 计时跑 `measurement_steps`（默认 10）个 step，把 step 总时长、compute、sync 分别存进三个列表。

#### 跨 rank 汇总（第 162–176 行）

把本 rank 的 `measurement_steps × 3` 时间矩阵打包成 tensor，用 `dist.all_gather` 收集所有 rank 的数据。

#### rank 0 统计与输出（第 178–235 行）

- **每次迭代取所有 rank 的最大值**：DDP 是同步训练，一步真正完成的时间取决于最慢的 rank。
- 计算均值/标准差，组装 `summary` 字典，核心指标：

  ```
  grad_sync_fraction = mean_grad_sync_ms / mean_step_ms
  ```

  即作业要求的「proportion of time spent on communicating gradients」。
- 终端打印汇总表；写两个文件到 `benchmark/<ddp_impl>_ddp_xl/`：
  - `summary.json`：配置 + 汇总指标（写报告直接引用）；
  - `results.csv`：每个迭代一行的明细（方便看抖动/画图）。

#### 清理（第 237–239 行）

barrier 后 `destroy_process_group()`，干净退出。

### 2.8 `main`（第 242–296 行）—— 入口和进程管理

- **参数定义**（第 243–261 行）：注意第 247–248 行用循环把 `XL_PRESET` 的每个键自动注册成 `--d_model`、`--num_layers` 等参数，默认值即 xl。
- **前置检查**（第 263–268 行）：GPU 数量、NCCL 可用性，不满足直接报错退出，而不是跑到一半崩掉。
- **Rendezvous**（第 270–275 行）：`--port 0` 时找一个空闲端口，写进 `MASTER_ADDR` / `MASTER_PORT` 环境变量——子进程里的 `init_process_group` 靠这两个变量找到彼此。
- **单进程模式**（第 278–280 行）：`world_size=1` 时不用多进程，直接跑 `worker(0, ...)`（单卡 baseline）。
- **多进程模式**（第 282–296 行）：`mp.spawn` 启动 `world_size` 个子进程各跑一份 `worker(rank, args)`。主进程每秒轮询，超过 `--timeout`（默认 30 分钟）抛超时；`finally` 块保证无论正常结束、超时还是异常，都先 `terminate` 再必要时 `kill` 所有子进程，**不会留下占着 GPU 的僵尸进程**。

---

## 3. 测量方法学小结

| 环节 | 手段 | 目的 |
|---|---|---|
| CUDA 异步 | 每个计时段前后 `torch.cuda.synchronize` | 秒表测到的是 GPU 真实完成时间 |
| 一次性开销 | 3 步 warmup 不计时 | 排除 kernel 编译、NCCL 建链 |
| rank 间偏差 | 每步前 `dist.barrier` | all_reduce 等待时间不污染通信计时 |
| 同步语义 | 每次迭代取跨 rank 最大值 | step 时间 = 最慢 rank 的时间 |
| 通信占比 | `sync / (compute + sync)` | 直接给出作业要的 proportion |

---

## 4. 常用命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--world_size` | 2 | 进程/GPU 数（1 节点 × 2 GPU） |
| `--micro_batch_size` | 4 | 每 GPU 每个微 batch 的序列数 |
| `--grad_accum_steps` | 8 | 每个 optimizer step 累积的微 batch 数 |
| `--mix_precision` | off | forward/loss 用 bf16 autocast |
| `--ddp_impl` | `naive` | `naive`：逐参数 all-reduce；`flat`：梯度扁平后单次 all-reduce；`none`：跳过同步（纯计算对照组） |
| `--warmup_steps` / `--measurement_steps` | 3 / 10 | 预热与正式测量步数 |
| `--output` | `benchmark/<ddp_impl>_ddp_xl` | 结果输出目录 |
| `--d_model` 等 | xl preset | 可覆盖模型规模 |

全局 batch 示例：默认 2 × 4 × 8 × 1024 = **65,536 tokens/step**；想接近 TinyStories 的 ~0.5M token，可调大 `--grad_accum_steps`。

---

## 5. 输出示例

终端：

```
=== Naive DDP benchmark (xl) ===
            world_size: 2
                   ddp: True
            num_params: 774094400
   global_batch_tokens: 65536
         mix_precision: True
             step time: xxxx.xx ± xx.xx ms
          compute time: xxxx.xx ms
        grad sync time:  xxx.xx ± xx.xx ms
    grad sync fraction: xx.xx%
```

文件（`benchmark/<ddp_impl>_ddp_xl/`）：

- `summary.json` — 全部配置 + `mean_step_ms`、`mean_grad_sync_ms`、`grad_sync_fraction` 等汇总指标；
- `results.csv` — 每迭代一行：`iteration, step_ms, compute_ms, grad_sync_ms, grad_sync_fraction`。
