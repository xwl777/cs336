# FlashAttention benchmark

默认配置已缩小为单张 RTX 5090 的起步测试范围。在 Linux + CUDA/Triton 环境，从 `assignment2-systems` 目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_flash_attention.py
```

默认固定 batch size 1、causal mask，遍历 N = 128、256、512、1024、2048、4096、8192，d = 16、32、64，以及 bfloat16、float32，共 42 组配置。每组比较两种实现的三个阶段，共 252 个结果。这个范围是保守起点，没有在 RTX 5090 上实测，仍会受到空闲显存及 kernel 编译资源的影响。

生成文件：

- `benchmark/flash_attention_5090.csv`：每行一个实现的一个阶段，记录延迟、状态、tile 大小与错误信息。
- `benchmark/flash_attention_5090.md`：每行一组配置，六列延迟，直接用于比较；未完成阶段显示 pending。
- `benchmark/flash_attention_5090.json`：GPU、显存容量、PyTorch/Triton/CUDA 版本和运行参数。

输入和随机上游梯度均在计时前生成，两种实现使用同一组输入。没有在本地生成或虚构任何 GPU 测量结果。

## 计时含义

使用 `triton.testing.do_bench`，默认 warmup=25 ms、rep=100 ms，报告平均毫秒数。这里的 rep 是计时预算，不是固定运行 100 次。每个阶段在调用 do_bench 前先执行一次并同步，使 JIT 编译和首次执行发生在测量之前。

- forward：启用梯度的训练前向，包括为反向保存数据的开销。
- backward：提前执行一次前向，测量 `torch.autograd.grad(..., retain_graph=True)`；前向不计时，重复复用同一计算图。
- forward_backward：每次重新前向并求输入梯度，不保留计算图；包括两个阶段，没有优化器。

使用 autograd.grad 避免 `.grad` 累加，每次返回 dQ/dK/dV。F+B 是直接测量，不能用单独 F 和 B 的测量值相加代替。do_bench 使用 CUDA 计时机制；不在被测函数内加入额外的 Python 同步操作。

PyTorch 基线显式执行矩阵乘法、causal mask、softmax、矩阵乘法，没有使用 SDPA、FlashAttention 或 torch.compile。Triton 实现已升级为 fused Triton forward + tiled Triton backward（可选题 Algorithm 2），JSON 中记录反向后端为 tiled_triton。它与先前 partially Triton 结果的反向实现不同，不应混为同一组结果。

## 精度、tile 与内存

PyTorch FP32 matmul 禁用 TF32；Triton FP32 dot 使用 tf32x3，它以多次 TF32 运算提高精度，不是严格 IEEE FP32 的逐操作一致比较。BF16 下，Triton 反向使用 BF16 dot 和 FP32 累加，D 的归约、概率和 dS 计算使用 FP32，在梯度 dot 前转换 P/dS 为输入 dtype，最终存储梯度时转换回输入 dtype。

默认 key tile 为 32，query tile 为 FP32 16、BF16 32；反向 query/key tile 为 16/32。降低 tile 可以减少单个 program 的资源压力，但不保证最优性能。可通过参数调整，每种配置的 tile 会记录在 CSV 中。例如：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_flash_attention.py \
  --seq-lens 128 256 512 1024 --d-models 16 32 \
  --q-tile-size 16 --k-tile-size 16 \
  --bwd-q-tile-size 16 --bwd-k-tile-size 16 \
  --output benchmark/flash_attention_small.csv
```

前向和反向都屏蔽尾块的无效 key，并对读写做边界检查。反向 tile 可独立用 `--bwd-q-tile-size`、`--bwd-k-tile-size` 设置；预检和正式计时共用同一套 tile 配置。

当前 Triton 反向只在局部块中重算 P/dS，没有完整 N × N 中间矩阵，也不使用原子累加。普通 PyTorch 基线仍可能 OOM。脚本对每个阶段独立记录 OOM，继续测其他阶段，保留已经成功的前向延迟；其他 RuntimeError 记录为 ERROR，细节见 CSV 和终端。不要把 OOM 或 ERROR 当成延迟为零。

Triton `OutOfResources` 单独记录为 RESOURCE_ERROR，表示 kernel 需要的资源超过硬件限制，不能直接解释为整张显卡显存耗尽。若 N=128、d=128、batch=1 都失败，张量本身很小，应检查完整错误是 CUDA OOM、shared memory 资源错误还是编译错误。仅缩短 N 不一定能解决单个 tile 的资源限制。

Triton 每种新签名的首次编译发生在计时前，不属于报告的延迟；它仍可能使整次实验耗时显著增加。

## GPU 上的建议执行顺序

先运行正确性测试：

```bash
uv run pytest -k 'test_flash_forward_pass_triton or test_flash_backward'
```

现有测试主要覆盖 float32。小规模检查两种精度能否完整运行：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_flash_attention.py \
  --seq-lens 128 --d-models 16 64 \
  --output benchmark/flash_attention_smoke.csv
```

脚本默认先在小输入上比较 O/dQ/dK/dV 与普通 attention 的结果，覆盖选择的每种 dtype 和 d；FP32 容差为 1e-2，BF16 为 5e-2。失败时停止，不会把错误实现的延迟当成有效结果。可用 `--skip-check` 跳过预检。预检只覆盖小尺寸，不代表所有长序列都经过正确性验证。

随后执行文档开头的缩小版 sweep。确保同一 GPU 上没有其他任务同时占用资源。现在默认允许任意 CUDA GPU，旧 `--allow-other-gpu` 参数保留兼容，不必再传。

需要试回 d=128 时，先单独测小尺寸与更小 tile：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_flash_attention.py \
  --seq-lens 128 256 --d-models 128 \
  --q-tile-size 16 --k-tile-size 16 \
  --bwd-q-tile-size 16 --bwd-k-tile-size 16 \
  --output benchmark/flash_attention_d128.csv
```

5090 的结果不能替代题目指定的单张 B200 结果。将来恢复作业完整规模时可以显式传入参数：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_flash_attention.py \
  --require-b200 \
  --seq-lens 128 256 512 1024 2048 4096 8192 16384 32768 65536 \
  --d-models 16 32 64 128 \
  --output benchmark/flash_attention_b200.csv
```

本次只进行了本地静态检查，没有运行 CUDA kernel、编译反向或 GPU benchmark；实际延迟和 GPU 正确性需要在目标机器上验证。
