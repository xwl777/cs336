"""Benchmark naive DDP (per-parameter gradient all-reduce) on 1 node x 2 GPUs.

Measures, for the xl model size (d_model=1600, num_heads=25, num_layers=24,
d_ff=6400, context_length=1024, vocab_size=10000) from Section 2.1.2:
  * total wall time per training step (forward + backward + grad sync + optimizer)
  * time spent communicating gradients inside finish_gradient_synchronization
  * the communication proportion of each step

Run from assignment2-systems:
    uv run python scripts/benchmark_naive_ddp.py
    uv run python scripts/benchmark_naive_ddp.py --micro_batch_size 2 --grad_accum_steps 16
    uv run python scripts/benchmark_naive_ddp.py --ddp_impl flat      # one batched all-reduce
    uv run python scripts/benchmark_naive_ddp.py --ddp_impl none      # compute-only reference

A training step processes a global batch of
    world_size * micro_batch_size * grad_accum_steps * context_length tokens
(default 2 * 4 * 8 * 1024 = 65536, close to the ~0.5M-token TinyStories batch
when --grad_accum_steps is raised). Batches are synthetic random tokens, since
we benchmark step timing, not convergence.
"""

import argparse
import csv
import json
import os
import socket
import statistics
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx

from cs336_basics.loss import cross_entropy
from cs336_basics.model import Transformer
from cs336_basics.optimizer import AdamW
from cs336_systems.flat_ddp import FlatDDP
from cs336_systems.naive_ddp import NaiveDDP
from cs336_systems.overlap_ddp import OverlapDDP

DDP_IMPLS = {"naive": NaiveDDP, "flat": FlatDDP, "overlap": OverlapDDP}

XL_PRESET = dict(
    d_model=1600,
    num_heads=25,
    num_layers=24,
    d_ff=6400,
    context_length=1024,
    vocab_size=10000,
    rope_theta=10000.0,
)


def build_model(args, device: torch.device) -> Transformer:
    return Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        device=device,
        dtype=torch.float32,
    )


def make_batch(args, device: torch.device):
    x = torch.randint(0, args.vocab_size, (args.micro_batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.micro_batch_size, args.context_length), device=device)
    return x, y


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx_range(enabled: bool, name: str):
    return nvtx.range(name) if enabled else nullcontext()


def train_step(model, ddp_model, optimizer, args, device, amp_context):
    """One full training step; returns (compute_seconds, grad_sync_seconds).

    compute covers forward + loss + backward over all micro-batches plus the
    optimizer step; grad_sync covers only finish_gradient_synchronization.
    """
    annotate = args.profile and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    synchronize(device)
    t_compute_start = time.perf_counter()

    for micro_batch in range(args.grad_accum_steps):
        x, y = make_batch(args, device)
        # For OverlapDDP: accumulate the first N-1 micro-batches locally, only
        # the last backward communicates (hooks fire but no_sync skips them).
        no_sync = (
            ddp_model.no_sync()
            if isinstance(ddp_model, OverlapDDP) and micro_batch < args.grad_accum_steps - 1
            else nullcontext()
        )
        with nvtx_range(annotate, f"micro_batch_{micro_batch}"):
            with no_sync:
                with nvtx_range(annotate, "forward"):
                    with amp_context:
                        logits = model(x)
                with nvtx_range(annotate, "loss"):
                    with amp_context:
                        loss = cross_entropy(logits=logits, targets=y) / args.grad_accum_steps
                with nvtx_range(annotate, "backward"):
                    loss.backward()

    synchronize(device)
    t_sync_start = time.perf_counter()

    if ddp_model is not None:
        with nvtx_range(annotate, "finish_gradient_sync"):
            ddp_model.finish_gradient_synchronization()

    synchronize(device)
    t_sync_end = time.perf_counter()

    with nvtx_range(annotate, "optimizer_step"):
        optimizer.step()

    synchronize(device)
    t_compute_end = time.perf_counter()

    compute_seconds = (t_compute_end - t_compute_start) - (t_sync_end - t_sync_start)
    sync_seconds = t_sync_end - t_sync_start
    return compute_seconds, sync_seconds


def worker(rank: int, args: argparse.Namespace) -> None:
    use_cuda = args.device == "cuda"
    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    if args.world_size > 1:
        dist.init_process_group(
            "nccl" if use_cuda else "gloo",
            rank=rank,
            world_size=args.world_size,
            timeout=timedelta(seconds=args.timeout),
        )

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    ddp_model = None
    if args.world_size > 1 and args.ddp_impl != "none":
        ddp_model = DDP_IMPLS[args.ddp_impl](model)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    amp_context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.mix_precision and use_cuda
        else nullcontext()
    )

    num_params = sum(p.numel() for p in model.parameters())
    grad_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    for _ in range(args.warmup_steps):
        train_step(model, ddp_model, optimizer, args, device, amp_context)
        if args.world_size > 1:
            dist.barrier()

    step_times, sync_times, compute_times = [], [], []
    annotate = args.profile and device.type == "cuda"
    for step_index in range(args.measurement_steps):
        if args.world_size > 1:
            dist.barrier()
            if use_cuda:
                torch.cuda.synchronize(device)
        start = time.perf_counter()
        with nvtx_range(annotate, f"step_{step_index}"):
            compute_s, sync_s = train_step(model, ddp_model, optimizer, args, device, amp_context)
        step_times.append(time.perf_counter() - start)
        compute_times.append(compute_s)
        sync_times.append(sync_s)

    # Aggregate per-step times across ranks: the step is only as fast as the
    # slowest rank, so report the max across ranks per iteration.
    if args.world_size > 1:
        local = torch.tensor(
            [[s, c, y] for s, c, y in zip(step_times, compute_times, sync_times)],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.empty_like(local) for _ in range(args.world_size)]
        dist.all_gather(gathered, local)
    else:
        gathered = [torch.tensor(
            [[s, c, y] for s, c, y in zip(step_times, compute_times, sync_times)],
            dtype=torch.float64,
        )]

    if rank == 0:
        per_rank = [t.cpu().tolist() if isinstance(t, torch.Tensor) else t.tolist() for t in gathered]
        elapsed = [[max(ranks[i][j] for ranks in per_rank) for j in range(3)] for i in range(args.measurement_steps)]
        steps = [row[0] for row in elapsed]
        computes = [row[1] for row in elapsed]
        syncs = [row[2] for row in elapsed]

        mean_step = statistics.mean(steps) * 1000
        mean_compute = statistics.mean(computes) * 1000
        mean_sync = statistics.mean(syncs) * 1000
        summary = {
            "preset": "xl",
            "world_size": args.world_size,
            "ddp_impl": args.ddp_impl,
            "device": args.device,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if use_cuda else ["cpu"],
            "num_params": num_params,
            "grad_bytes_per_rank": grad_bytes,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "context_length": args.context_length,
            "global_batch_tokens": args.world_size * args.micro_batch_size * args.grad_accum_steps * args.context_length,
            "mix_precision": args.mix_precision,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "mean_step_ms": mean_step,
            "std_step_ms": statistics.stdev(steps) * 1000 if len(steps) > 1 else 0.0,
            "mean_compute_ms": mean_compute,
            "mean_grad_sync_ms": mean_sync,
            "std_grad_sync_ms": statistics.stdev(syncs) * 1000 if len(syncs) > 1 else 0.0,
            "grad_sync_fraction": mean_sync / mean_step if mean_step > 0 else 0.0,
            "torch_version": torch.__version__,
        }

        print(f"\n=== DDP benchmark ({args.ddp_impl}) ===")
        for key in ("world_size", "ddp_impl", "num_params", "global_batch_tokens", "mix_precision"):
            print(f"{key:>22}: {summary[key]}")
        print(f"{'step time':>22}: {mean_step:.2f} ± {summary['std_step_ms']:.2f} ms")
        print(f"{'compute time':>22}: {mean_compute:.2f} ms")
        print(f"{'grad sync time':>22}: {mean_sync:.2f} ± {summary['std_grad_sync_ms']:.2f} ms")
        print(f"{'grad sync fraction':>22}: {summary['grad_sync_fraction'] * 100:.2f}%")

        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        fieldnames = ["iteration", "step_ms", "compute_ms", "grad_sync_ms", "grad_sync_fraction"]
        with (args.output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for i, (s, c, y) in enumerate(zip(steps, computes, syncs)):
                writer.writerow({
                    "iteration": i,
                    "step_ms": s * 1000,
                    "compute_ms": c * 1000,
                    "grad_sync_ms": y * 1000,
                    "grad_sync_fraction": y / s if s > 0 else 0.0,
                })
        print(f"\nSaved summary.json and results.csv to {args.output}")

    if args.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world_size", type=int, default=2, help="Processes/GPUs (1 node x 2 GPUs default).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    for key, value in XL_PRESET.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)

    parser.add_argument("--micro_batch_size", type=int, default=4, help="Sequences per GPU per micro-batch.")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Micro-batches per optimizer step.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mix_precision", action="store_true", help="bf16 autocast for forward/loss.")
    parser.add_argument(
        "--ddp_impl",
        choices=["naive", "flat", "overlap", "none"],
        default="naive",
        help="naive: per-parameter all-reduce; flat: one all-reduce of the concatenated "
        "gradients; overlap: async per-parameter all-reduce during backward; "
        "none: skip gradient sync (compute-only reference).",
    )
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit NVTX ranges (step/micro_batch/forward/backward/finish_gradient_sync/optimizer_step) "
        "for Nsight Systems. Run under: nsys profile --trace=cuda,nvtx,osrt -o <name> ...",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-run wall-time limit in seconds.")
    parser.add_argument("--port", type=int, default=0, help="Rendezvous port; 0 selects a free local port.")
    parser.add_argument("--output", type=Path, default=None, help="Default: benchmark/<ddp_impl>_ddp_xl")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"benchmark/{args.ddp_impl}_ddp_xl")

    if args.world_size < 1:
        parser.error("--world_size must be >= 1")
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size):
        parser.error(f"This benchmark needs {args.world_size} visible CUDA GPUs; use CUDA_VISIBLE_DEVICES to select them.")
    if args.device == "cuda" and args.world_size > 1 and not dist.is_nccl_available():
        parser.error("This PyTorch installation does not support NCCL.")

    if args.port == 0:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            args.port = sock.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.port)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.world_size == 1:
        worker(0, args)
        return

    start = time.monotonic()
    context = mp.spawn(worker, args=(args,), nprocs=args.world_size, join=False)
    try:
        while not context.join(timeout=1):
            if time.monotonic() - start >= args.timeout:
                raise TimeoutError(f"Benchmark exceeded {args.timeout:g}s; partial output may remain in {args.output}.")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)


if __name__ == "__main__":
    main()
