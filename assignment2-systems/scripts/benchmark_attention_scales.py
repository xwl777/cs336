import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def synchronize() -> None:
    torch.cuda.synchronize()


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bqk,bkd->bqd", probs, v)


def make_inputs(batch_size: int, seq_len: int, d_model: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, seq_len, d_model)
    q = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    dout = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v, dout


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def mean_ms(times: list[float]) -> float:
    return 1000.0 * statistics.mean(times)


def std_ms(times: list[float]) -> float:
    return 1000.0 * statistics.stdev(times) if len(times) > 1 else 0.0


def run_config(seq_len: int, d_model: int, args: argparse.Namespace) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        q, k, v, dout = make_inputs(BATCH_SIZE, seq_len, d_model, args.dtype)
        synchronize()

        for _ in range(args.warmup):
            out = attention(q, k, v)
            synchronize()
            out.backward(dout)
            synchronize()
            clear_grads(q, k, v)
            del out

        forward_times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            out = attention(q, k, v)
            synchronize()
            forward_times.append(time.perf_counter() - start)
            del out

        backward_times = []
        memory_before_backward = []
        for _ in range(args.iters):
            clear_grads(q, k, v)
            out = attention(q, k, v)
            synchronize()
            memory_before_backward.append(torch.cuda.memory_allocated())

            start = time.perf_counter()
            out.backward(dout)
            synchronize()
            backward_times.append(time.perf_counter() - start)
            del out

        peak_memory = torch.cuda.max_memory_allocated()
        result = {
            "batch_size": BATCH_SIZE,
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": str(args.dtype).removeprefix("torch."),
            "status": "ok",
            "forward_mean_ms": mean_ms(forward_times),
            "forward_std_ms": std_ms(forward_times),
            "backward_mean_ms": mean_ms(backward_times),
            "backward_std_ms": std_ms(backward_times),
            "memory_before_backward_mib": statistics.mean(memory_before_backward) / 2**20,
            "peak_memory_mib": peak_memory / 2**20,
            "error": "",
        }
        del q, k, v, dout
        torch.cuda.empty_cache()
        return result
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {
            "batch_size": BATCH_SIZE,
            "seq_len": seq_len,
            "d_model": d_model,
            "dtype": str(args.dtype).removeprefix("torch."),
            "status": "oom",
            "forward_mean_ms": "",
            "forward_std_ms": "",
            "backward_mean_ms": "",
            "backward_std_ms": "",
            "memory_before_backward_mib": "",
            "peak_memory_mib": "",
            "error": str(exc).splitlines()[0],
        }


def parse_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument("--output", type=Path, default=Path("benchmark/attention_scales.csv"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "seq_len",
        "d_model",
        "dtype",
        "status",
        "forward_mean_ms",
        "forward_std_ms",
        "backward_mean_ms",
        "backward_std_ms",
        "memory_before_backward_mib",
        "peak_memory_mib",
        "error",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                result = run_config(seq_len, d_model, args)
                writer.writerow(result)
                f.flush()
                print(result)


if __name__ == "__main__":
    main()
