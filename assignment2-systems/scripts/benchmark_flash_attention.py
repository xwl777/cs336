"""Single-GPU causal attention benchmark; see benchmark/FLASH_BENCHMARK.md."""

import argparse
import csv
import gc
import itertools
import json
import math
from pathlib import Path

import torch
import triton
import triton.testing
from triton.runtime.errors import OutOfResources

from cs336_systems.flash_attention import FlashAttentionTriton


PHASES = ("forward", "backward", "forward_backward")
PROVIDERS = ("pytorch", "triton")
FIELDS = (
    "seq_len", "d_model", "dtype", "implementation", "phase", "status", "latency_ms",
    "q_tile_size", "k_tile_size", "bwd_q_tile_size", "bwd_k_tile_size", "error",
)


def pytorch_attention(q, k, v):
    # Explicit eager operations: do not dispatch to SDPA or FlashAttention.
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    positions = torch.arange(q.shape[-2], device=q.device)
    mask = positions[:, None] >= positions[None, :]
    scores = scores.masked_fill(~mask, -1e6)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def flash_attention(q, k, v):
    return FlashAttentionTriton.apply(q, k, v, True)


def configure_tiles(args, dtype_name):
    q_tile = args.q_tile_size or (16 if dtype_name == "float32" else 32)
    FlashAttentionTriton.Q_TILE_SIZE = q_tile
    FlashAttentionTriton.K_TILE_SIZE = args.k_tile_size
    FlashAttentionTriton.BWD_Q_TILE_SIZE = args.bwd_q_tile_size
    FlashAttentionTriton.BWD_K_TILE_SIZE = args.bwd_k_tile_size
    return q_tile


def check_correctness(args):
    # Validate both precisions on small inputs before an expensive full sweep.
    for d, dtype_name in itertools.product(args.d_models, args.dtypes):
        configure_tiles(args, dtype_name)
        n = max(128, args.k_tile_size)
        dtype = getattr(torch, dtype_name)
        inputs = tuple(torch.randn((1, n, d), device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
        grad_o = torch.randn_like(inputs[0])
        reference = pytorch_attention(*inputs)
        actual = flash_attention(*inputs)
        expected_grads = torch.autograd.grad(reference, inputs, grad_o)
        actual_grads = torch.autograd.grad(actual, inputs, grad_o)
        tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-2
        for name, result, expected in zip(("O", "dQ", "dK", "dV"), (actual, *actual_grads), (reference, *expected_grads)):
            torch.testing.assert_close(result, expected, rtol=tolerance, atol=tolerance, msg=lambda msg: f"{dtype_name} d={d} {name}: {msg}")
        torch.cuda.synchronize()
        print(f"Correctness passed: {dtype_name} d={d} N={n}", flush=True)


def time_phase(fn, inputs, grad_o, phase, args):
    if phase == "forward":
        call = lambda: fn(*inputs)
    elif phase == "backward":
        # Forward setup is outside timing. Reuse its saved tensors on every run.
        output = fn(*inputs)
        call = lambda: torch.autograd.grad(output, inputs, grad_o, retain_graph=True)
    else:
        def call():
            output = fn(*inputs)
            return torch.autograd.grad(output, inputs, grad_o)

    # Compile and execute once before do_bench's own warmup and timing.
    call()
    torch.cuda.synchronize()
    return triton.testing.do_bench(call, warmup=args.warmup_ms, rep=args.rep_ms, return_mode="mean")


def measure(fn, inputs, grad_o, phase, args):
    try:
        latency = time_phase(fn, inputs, grad_o, phase, args)
        torch.cuda.synchronize()
        return {"status": "ok", "latency_ms": latency, "error": ""}
    except torch.cuda.OutOfMemoryError as exc:
        return {"status": "oom", "latency_ms": "", "error": str(exc).splitlines()[0]}
    except OutOfResources as exc:
        return {"status": "resource_error", "latency_ms": "", "error": str(exc)}
    except RuntimeError as exc:
        return {"status": "error", "latency_ms": "", "error": str(exc)}


def write_table(path, rows, gpu_name):
    indexed = {
        (r["seq_len"], r["d_model"], r["dtype"], r["implementation"], r["phase"]): r
        for r in rows
    }
    lines = [
        "# Causal Attention Benchmark", "",
        f"GPU: {gpu_name}. Batch size: 1. Mean latency in milliseconds.", "",
        "Triton implementation: fused Triton forward + tiled Triton backward.",
        "Backward reuses a retained forward graph; forward-backward builds a fresh graph each call.", "",
        "| N | d | dtype | PyTorch F | Triton F | PyTorch B | Triton B | PyTorch F+B | Triton F+B |",
        "|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    configs = dict.fromkeys((r["seq_len"], r["d_model"], r["dtype"]) for r in rows)
    for config in configs:
        cells = [str(item) for item in config]
        for phase in PHASES:
            for provider in PROVIDERS:
                row = indexed.get((*config, provider, phase))
                if row is None:
                    cells.append("pending")
                elif row["status"] == "ok":
                    cells.append(f"{row['latency_ms']:.4f}")
                else:
                    cells.append(row["status"].upper())
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2**i for i in range(7, 14)])
    parser.add_argument("--d-models", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--dtypes", nargs="+", choices=["bfloat16", "float32"], default=["bfloat16", "float32"])
    parser.add_argument("--q-tile-size", type=int, default=None, help="Default: 16 for FP32, 32 for BF16")
    parser.add_argument("--k-tile-size", type=int, default=32)
    parser.add_argument("--bwd-q-tile-size", type=int, default=16)
    parser.add_argument("--bwd-k-tile-size", type=int, default=32)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("benchmark/flash_attention_5090.csv"))
    parser.add_argument("--require-b200", action="store_true", help="Require a B200 for assignment submission runs")
    parser.add_argument("--allow-other-gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-check", action="store_true", help="Skip the small forward/gradient correctness preflight")
    args = parser.parse_args()
    sizes = args.seq_lens + args.d_models + [args.k_tile_size, args.bwd_q_tile_size, args.bwd_k_tile_size]
    if args.q_tile_size is not None:
        sizes.append(args.q_tile_size)
    if any(n < 16 or n & (n - 1) for n in sizes):
        parser.error("Sequence lengths, embedding dimensions and tile sizes must be powers of two >= 16.")
    if args.warmup_ms <= 0 or args.rep_ms <= 0:
        parser.error("Warmup and repetition durations must be positive.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this script on a CUDA GPU with Triton installed.")
    # CUDA_VISIBLE_DEVICES can select the physical GPU; use only cuda:0 here.
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_properties(0)
    if args.require_b200 and "B200" not in gpu.name.upper():
        raise RuntimeError(f"--require-b200 was set, but detected {gpu.name}.")
    print(f"GPU: {gpu.name}; batch=1; N={args.seq_lens}; d={args.d_models}; dtype={args.dtypes}", flush=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not args.skip_check:
        check_correctness(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_path = args.output.with_suffix(".md")
    metadata = {
        "gpu": gpu.name, "gpu_memory_bytes": gpu.total_memory, "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda,
        "batch_size": 1, "is_causal": True, "statistic": "mean", "unit": "ms",
        "pytorch_matmul_precision": "highest", "triton_float32_dot_precision": "tf32x3",
        "triton_backward_backend": "tiled_triton",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    rows = []
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for n, d, dtype_name in itertools.product(args.seq_lens, args.d_models, args.dtypes):
            gc.collect()
            torch.cuda.empty_cache()
            q_tile = configure_tiles(args, dtype_name)
            inputs = grad_o = None
            allocation_error = None
            try:
                dtype = getattr(torch, dtype_name)
                inputs = tuple(torch.randn((1, n, d), device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
                grad_o = torch.randn((1, n, d), device="cuda", dtype=dtype)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as exc:
                allocation_error = str(exc).splitlines()[0]

            for provider, fn in (("pytorch", pytorch_attention), ("triton", flash_attention)):
                for phase in PHASES:
                    if allocation_error is not None:
                        result = {"status": "oom", "latency_ms": "", "error": allocation_error}
                    else:
                        result = measure(fn, inputs, grad_o, phase, args)
                    row = {
                        "seq_len": n, "d_model": d, "dtype": dtype_name,
                        "implementation": provider, "phase": phase,
                        "q_tile_size": q_tile if provider == "triton" else "",
                        "k_tile_size": args.k_tile_size if provider == "triton" else "",
                        "bwd_q_tile_size": args.bwd_q_tile_size if provider == "triton" else "",
                        "bwd_k_tile_size": args.bwd_k_tile_size if provider == "triton" else "", **result,
                    }
                    rows.append(row)
                    writer.writerow(row)
                    file.flush()
                    write_table(table_path, rows, gpu.name)
                    print(f"N={n} d={d} {dtype_name} {provider} {phase}: {result['latency_ms']} {result['status']}", flush=True)
                    if result["status"] != "ok":
                        print(result["error"], flush=True)
                    # measure has returned: its retained graph and exception
                    # traceback no longer keep large attention buffers alive.
                    gc.collect()
                    torch.cuda.empty_cache()
            del inputs, grad_o
    print(f"Results: {args.output}\nTable: {table_path}")


if __name__ == "__main__":
    main()
