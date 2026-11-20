"""Benchmark float32 NCCL all-reduce on exactly two local GPUs.

Run from assignment2-systems:
    uv run python scripts/benchmark_all_reduce.py
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/benchmark_all_reduce.py

Sizes use decimal units: 1 MB = 1,000,000 bytes, 1 GB = 1,000,000,000
bytes, per rank. Timings include host launch and CUDA completion, but exclude
allocation, correctness checks, warmup, and the pre-iteration barrier.
"""

import argparse
import csv
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


WORLD_SIZE = 2
SIZES = [("1MB", 1_000_000), ("10MB", 10_000_000), ("100MB", 100_000_000), ("1GB", 1_000_000_000)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker(rank: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=WORLD_SIZE,
        timeout=timedelta(seconds=args.timeout),
    )
    summaries, samples = [], []
    try:
        for label, size_bytes in SIZES:
            # Check allocations collectively so one rank's OOM cannot strand its peer.
            data = None
            allocated = torch.ones(1, dtype=torch.int32, device=device)
            try:
                data = torch.full((size_bytes // 4,), rank + 1.0, dtype=torch.float32, device=device)
            except torch.cuda.OutOfMemoryError:
                allocated.zero_()
            if not allocated.item():
                raise RuntimeError(f"Insufficient GPU memory for {label} on at least one rank.")

            # SUM must produce 1 + 2 = 3 on both ranks; this is outside timing.
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            correct = torch.all(data == 3).to(dtype=torch.int32).reshape(1)
            dist.all_reduce(correct, op=dist.ReduceOp.MIN)
            if not correct.item():
                raise RuntimeError(f"All-reduce correctness check failed for {label}.")
            # Zeros remain finite after repeated in-place sums, without timed resets.
            data.zero_()
            for _ in range(args.warmup):
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
                torch.cuda.synchronize(device)

            local_ms = []
            for _ in range(args.iters):
                dist.barrier()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
                torch.cuda.synchronize(device)
                local_ms.append((time.perf_counter() - start) * 1000)

            # Gather after measurement; reporting collectives never enter the timer.
            timing = torch.tensor(local_ms, dtype=torch.float64, device=device)
            gathered = [torch.empty_like(timing) for _ in range(WORLD_SIZE)]
            dist.all_gather(gathered, timing)
            if rank == 0:
                rank_times = [tensor.cpu().tolist() for tensor in gathered]
                elapsed = [max(pair) for pair in zip(*rank_times)]
                mean = statistics.mean(elapsed)
                row = {
                    "size": label,
                    "size_bytes_per_rank": size_bytes,
                    "numel_per_rank": size_bytes // 4,
                    "world_size": WORLD_SIZE,
                    "dtype": "float32",
                    "backend": "nccl",
                    "warmup": args.warmup,
                    "iterations": args.iters,
                    "mean_ms": mean,
                    "median_ms": statistics.median(elapsed),
                    "std_ms": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
                    "min_ms": min(elapsed),
                    "max_ms": max(elapsed),
                    # Payload / time; not the aggregate traffic across both GPUs.
                    "algorithm_bandwidth_GB_s": size_bytes / (mean / 1000) / 1e9,
                }
                summaries.append(row)
                for index, duration in enumerate(elapsed):
                    samples.append({
                        "size": label, "iteration": index,
                        "rank0_ms": rank_times[0][index],
                        "rank1_ms": rank_times[1][index], "max_rank_ms": duration,
                    })
                # Preserve completed configurations even if a later size fails.
                write_csv(args.output / "results.csv", summaries)
                write_csv(args.output / "samples.csv", samples)
                print(f"{label:>5} | 2 GPUs | mean {mean:.3f} ms | "
                      f"std {row['std_ms']:.3f} ms | {row['algorithm_bandwidth_GB_s']:.2f} GB/s", flush=True)
            del data, timing, gathered, allocated, correct
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


def plot_results(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sizes = [float(row["size_bytes_per_rank"]) / 1e6 for row in rows]
    means = [float(row["mean_ms"]) for row in rows]
    stds = [float(row["std_ms"]) for row in rows]
    bandwidths = [float(row["algorithm_bandwidth_GB_s"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].errorbar(sizes, means, yerr=stds, marker="o", capsize=4)
    axes[0].set_ylabel("Mean all-reduce latency (ms), ±1 sample std")
    axes[1].plot(sizes, bandwidths, marker="o")
    axes[1].set_ylabel("Algorithm bandwidth (GB/s): payload / time")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(sizes, [row["size"] for row in rows])
        axis.set_xlabel("Float32 tensor size per GPU (decimal units)")
        axis.grid(True, alpha=0.3)
    fig.suptitle("NCCL SUM all-reduce · 2 GPUs / 2 processes")
    fig.savefig(output / "all_reduce.png", dpi=180)
    plt.close(fig)
    commentary = (
        f"With two GPUs, mean latency was {means[0]:.3f} ms for 1MB and {means[-1]:.3f} ms for 1GB; "
        f"payload/time bandwidth was {bandwidths[0]:.2f} and {bandwidths[-1]:.2f} GB/s, respectively. "
        "Small payloads are sensitive to fixed launch and synchronization costs, while larger payloads "
        "place more demand on the GPU interconnect; GPU load and interconnect topology also affect these measurements. "
        "The process count is fixed at two, so this experiment does not measure scaling to four or six GPUs.\n"
    )
    (output / "commentary.txt").write_text(commentary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=240, help="Worker wall-time limit in seconds (maximum 270).")
    parser.add_argument("--port", type=int, default=0, help="Rendezvous port; 0 selects an available local port.")
    parser.add_argument("--output", type=Path, default=Path("benchmark/all_reduce_2gpu"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if not 0 < args.timeout <= 270:
        parser.error("--timeout must be in (0, 270] seconds")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        parser.error("This benchmark requires at least two visible CUDA GPUs. Use CUDA_VISIBLE_DEVICES to select them.")
    if not dist.is_nccl_available():
        parser.error("This PyTorch installation does not support NCCL.")
    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(), "world_size": WORLD_SIZE,
        "gpus": [torch.cuda.get_device_name(i) for i in range(WORLD_SIZE)],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warmup": args.warmup, "iterations": args.iters, "timeout_s": args.timeout,
        "size_units": "decimal MB/GB, per rank", "timing": "max rank wall time per iteration, synchronized CUDA",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    context = mp.spawn(worker, args=(args,), nprocs=WORLD_SIZE, join=False)
    try:
        while not context.join(timeout=1):
            if time.monotonic() - start >= args.timeout:
                raise TimeoutError(f"Benchmark exceeded {args.timeout:g}s; completed rows remain in {args.output}.")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
    plot_results(args.output)
    print(f"Saved results.csv, samples.csv, metadata.json, all_reduce.png and commentary.txt to {args.output}")


if __name__ == "__main__":
    main()
