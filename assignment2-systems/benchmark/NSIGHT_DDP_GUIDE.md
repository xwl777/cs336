# Nsight Systems 对比 naive vs overlap DDP 操作指南

对应作业：**ddp_overlap_individual_parameters_benchmarking (b)** —— 用 Nsight profiler 插桩，
截取两张 trace 图，直观展示"naive 的通信不与 backward 重叠，overlap 的重叠"。

前置：[`OVERLAP_DDP_GUIDE.md`](./OVERLAP_DDP_GUIDE.md)（三种 DDP 实现）、
GPU 机器上已装 `nsys`（Nsight Systems CLI，随 CUDA toolkit 提供）。

---

## 1. benchmark 脚本已内置的插桩

`scripts/benchmark_naive_ddp.py` 加了 `--profile` 开关（仅 CUDA 下生效），开启后发射以下 NVTX range：

```
step_{i}                        每个测量 step
└─ micro_batch_{j}              每个 micro-batch
   ├─ forward
   ├─ loss
   └─ backward                  ← overlap 的 all-reduce 就发生在这段期间
├─ finish_gradient_sync         ← naive/flat 的通信全部堆在这段
└─ optimizer_step
```

另外你的 `cs336_basics` 模型内部本来就有 NVTX 注解
（`scaled dot product attention`、`computing attention scores` 等），也会出现在 trace 里。

## 2. 采集命令（GPU 机器上，assignment2-systems 目录下）

```bash
# ① naive（初始 DDP 实现，通信不重叠）
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  -o bench_naive \
  uv run python scripts/benchmark_naive_ddp.py \
      --ddp_impl naive --mix_precision --profile \
      --warmup_steps 2 --measurement_steps 3

# ② overlap（通信与 backward 重叠）
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  -o bench_overlap \
  uv run python scripts/benchmark_naive_ddp.py \
      --ddp_impl overlap --mix_precision --profile \
      --warmup_steps 2 --measurement_steps 3
```

说明：
- `mp.spawn` 出的两个 rank 进程会被 nsys 一起跟踪，生成 `bench_naive.nsys-rep`（含两个进程的 timeline）；
- `--measurement_steps 3` 足够截图，trace 文件也小；
- 若机器装了 Nsight Systems GUI，直接打开 `.nsys-rep`；纯命令行可用
  `nsys stats bench_naive.nsys-rep` 看汇总表（`cuda_gpu_kern_sum` 里 NCCL kernel 的总时长）。

## 3. 截图要点（deliverable 的两张图）

打开 trace 后，展开 rank 进程的 CUDA HW 行，会看到多条 stream：
**计算 stream**（默认 stream，matmul/backward kernel）和 **NCCL 通信 stream**
（kernel 名以 `ncclDevKernel_AllReduce` 开头）。用 NVTX 行（`step_i` / `backward` /
`finish_gradient_sync`）对齐时间轴，然后放大**一个 step**：

### 图 1：naive（应展示"不重叠"）

预期看到的画面：

```
NVTX:      [-------- backward --------][finish_gradient_sync][optimizer_step]
计算stream: ████████████████████████████│                    │██
NCCL stream:                            │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
                                        └─ 300+ 个 AllReduce kernel 密集排在
                                           backward 结束之后，期间计算 stream 空闲
```

截图关键特征：**NCCL kernel 全部位于 `backward` NVTX 段之后**，且与计算 kernel 在时间轴上**不交叠**。

### 图 2：overlap（应展示"重叠"）

```
NVTX:      [-------- backward --------][sync][optimizer_step]
计算stream: ████████████████████████████│    │██
NCCL stream:   ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓ ▓▓▓▓│    │
               └─ AllReduce kernel 与 backward 计算 kernel 交错并行，
                  finish_gradient_sync 段几乎为空
```

截图关键特征：**NCCL kernel 穿插在 `backward` NVTX 段内部**、与计算 kernel 同一时间窗口，
`finish_gradient_sync` 段几乎瞬时。

两张图建议相同缩放比例、相同 step 位置截取，对比才有说服力。

## 4. (a) 部分的计时对比命令

```bash
for impl in none naive flat overlap; do
  CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/benchmark_naive_ddp.py \
      --ddp_impl $impl --mix_precision
done
```

结果在 `benchmark/<impl>_ddp_xl/summary.json`，对比字段：
`mean_step_ms`（deliverable 要的 time per training iteration）、
`mean_grad_sync_ms`、`grad_sync_fraction`。

预期结论模板（拿到真实数字后替换 X/Y/Z）：

> 逐参数 all-reduce 的 naive DDP 每步 X ms（通信占 ~A%）；扁平化单次 all-reduce 降到 Y ms（通信 ~B%）；
> 用 post-accumulate hook 将逐参数通信与 backward 重叠后，每步 Z ms，逼近单卡纯计算对照组（none），
> 同步收尾时间趋近 0——通信开销基本被 backward 计算掩盖。

## 5. 常见问题

- **trace 里看不到 NVTX 行**：确认加了 `--profile` 且 `--device cuda`（CPU 下注解自动关闭）；
  `--trace` 里必须含 `nvtx`。
- **看不到 NCCL kernel**：`--trace` 必须含 `cuda`；确认两个 rank 真的在跑
  （日志里有 world_size: 2）。
- **`.nsys-rep` 太大**：减小 `--measurement_steps` / `--warmup_steps`，
  或加 `--duration=120` 限时。
- **两个 rank 的时间轴对不上**：以 NVTX `step_i` 为准对齐；rank 间有 barrier，起点基本一致。
