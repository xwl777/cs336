"""Profile optimizer state sharding (default: one node, two GPUs, xl).

Run: uv run python scripts/benchmark_sharded_optimizer.py --mix_precision
Estimate without CUDA: add --estimate_only.
CPU smoke: --device cpu --d_model 16 --num_heads 2 --d_ff 32 --num_layers 2
           --context_length 8 --vocab_size 32 --micro_batch_size 2 --grad_accum_steps 2
"""
import argparse
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

from benchmark_naive_ddp import XL_PRESET, build_model, make_batch, synchronize, train_step
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.overlap_ddp import OverlapDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


def tensor_bytes(value, device_type):
    """Count unique tensor storages in a nested structure on the target device."""
    seen = set()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != device_type:
                return 0
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (tuple, list)):
            return sum(visit(v) for v in item)
        return 0
    return visit(value)


def snapshot(model, optimizer, device, label):
    synchronize(device)
    parts = {
        "parameters_bytes": tensor_bytes(list(model.parameters()), device.type),
        "gradients_bytes": tensor_bytes([p.grad for p in model.parameters()], device.type),
        "optimizer_states_bytes": tensor_bytes(dict(optimizer.state), device.type),
        "buffers_bytes": tensor_bytes(list(model.buffers()), device.type),
    }
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        parts.update(allocated_bytes=allocated,
                     reserved_bytes=torch.cuda.memory_reserved(device),
                     phase_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                     phase_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                     other_live_bytes=allocated - sum(parts.values()))
    else:
        # CPU tensor accounting is not a CUDA or CPU peak memory measurement.
        parts.update(allocated_bytes=None, reserved_bytes=None,
                     phase_peak_allocated_bytes=None, phase_peak_reserved_bytes=None,
                     other_live_bytes=None)
    return {"checkpoint": label, **parts}


def reset_peak(device):
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def memory_step(model, optimizer, args, device, amp):
    optimizer.zero_grad(set_to_none=True)
    reset_peak(device)
    for index in range(args.grad_accum_steps):
        context = model.no_sync() if index < args.grad_accum_steps - 1 else nullcontext()
        with context:
            x, y = make_batch(args, device)
            with amp():
                logits = model(x)
                loss = cross_entropy(logits=logits, targets=y) / args.grad_accum_steps
            loss.backward()
        del x, y, logits, loss
    model.finish_gradient_synchronization()
    before = snapshot(model, optimizer, device, "before_optimizer_step")
    reset_peak(device)
    optimizer.step()
    after = snapshot(model, optimizer, device, "after_optimizer_step")
    return {"before": before, "after": after}


def worker(rank, args, mode):
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo",
                            rank=rank, world_size=args.world_size,
                            timeout=timedelta(seconds=args.timeout))
    try:
        torch.manual_seed(args.seed)
        reset_peak(device)
        model = OverlapDDP(build_model(args, device))
        optimizer = (ShardedOptimizer(model.parameters(), AdamW, lr=args.lr)
                     if mode == "sharded" else AdamW(model.parameters(), lr=args.lr))
        initial = snapshot(model, optimizer, device, "after_initialization")
        def amp():
            return (torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.mix_precision and args.device == "cuda" else nullcontext())
        first = memory_step(model, optimizer, args, device, amp)
        peak_so_far = 0
        if device.type == "cuda":
            for point in (initial, first["before"], first["after"]):
                peak_so_far = max(peak_so_far, point["phase_peak_allocated_bytes"])
                point["peak_since_initialization_bytes"] = peak_so_far
        for _ in range(args.warmup_steps):
            train_step(model, model, optimizer, args, device, amp())
        if device.type == "cuda":
            peak_so_far = max(peak_so_far, torch.cuda.max_memory_allocated(device))
        steady = memory_step(model, optimizer, args, device, amp)
        if device.type == "cuda":
            for point in (steady["before"], steady["after"]):
                peak_so_far = max(peak_so_far, point["phase_peak_allocated_bytes"])
                point["peak_since_initialization_bytes"] = peak_so_far
        # Time complete iterations independently of memory instrumentation.
        times = []
        for _ in range(args.measurement_steps):
            synchronize(device)
            dist.barrier()
            synchronize(device)
            start = time.perf_counter()
            train_step(model, model, optimizer, args, device, amp())
            times.append(time.perf_counter() - start)
        record = {"rank": rank, "initialization": initial,
                  "first_step": first, "steady_step": steady, "step_seconds": times}
        gathered = [None] * args.world_size
        dist.all_gather_object(gathered, record)
        if rank == 0:
            slowest = [max(row["step_seconds"][i] for row in gathered)
                       for i in range(args.measurement_steps)]
            summary = {
                "mode": mode, "device": args.device, "world_size": args.world_size,
                "torch_version": torch.__version__,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(args.world_size)]
                             if args.device == "cuda" else [],
                "model_config": {key: getattr(args, key) for key in XL_PRESET},
                "num_params": sum(p.numel() for p in model.parameters()),
                "optimizer_cls": "cs336_basics.optimizer.AdamW",
                "parameter_dtype": "float32", "bf16_autocast": args.mix_precision,
                "micro_batch_size": args.micro_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "global_batch_tokens": args.world_size * args.micro_batch_size * args.grad_accum_steps * args.context_length,
                "warmup_steps": args.warmup_steps, "measurement_steps": args.measurement_steps,
                "mean_step_ms": statistics.mean(slowest) * 1000,
                "std_step_ms": statistics.stdev(slowest) * 1000 if len(slowest) > 1 else 0,
                "ranks": gathered,
            }
            path = args.output / f"{mode}.json"
            path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{mode}: {summary['mean_step_ms']:.2f} ms/iteration; saved {path}", flush=True)
    finally:
        dist.destroy_process_group()


def estimate(args):
    model = build_model(args, torch.device("meta"))
    sizes = [p.numel() for p in model.parameters()]
    n = sum(sizes)
    p = 4 * n
    result = {"kind": "analytic_model_state_storage_NOT_measured_peak",
              "num_params": n, "parameter_dtype": "float32",
              "parameters_bytes": p, "gradients_bytes": p,
              "initialization_model_state_bytes": p,
              "first_before_optimizer_model_state_bytes": 2 * p,
              "unsharded_optimizer_state_bytes": 2 * p,
              "unsharded_after_and_steady_model_state_bytes": 4 * p,
              "ranks": [{"rank": rank,
                         "owned_numel": sum(sizes[rank::args.world_size]),
                         "sharded_optimizer_state_bytes": 8 * sum(sizes[rank::args.world_size]),
                         "sharded_after_and_steady_model_state_bytes": 2 * p + 8 * sum(sizes[rank::args.world_size])}
                        for rank in range(args.world_size)]}
    path = args.output / "estimate.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in XL_PRESET.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--mode", choices=["both", "unsharded", "sharded"], default="both")
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument("--mix_precision", action="store_true")
    parser.add_argument("--estimate_only", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=Path("benchmark/sharded_optimizer_xl"))
    args = parser.parse_args()
    args.profile = False
    for name in ("world_size", "micro_batch_size", "grad_accum_steps", "measurement_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup_steps < 0 or args.timeout <= 0:
        parser.error("warmup_steps must be nonnegative and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.estimate_only:
        estimate(args)
        return
    if args.device == "cuda" and (torch.cuda.device_count() < args.world_size or not dist.is_nccl_available()):
        parser.error(f"Need {args.world_size} CUDA GPUs and NCCL; use --estimate_only for storage estimates or --device cpu with a tiny model for smoke testing")
    for mode in (["unsharded", "sharded"] if args.mode == "both" else [args.mode]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        context = mp.spawn(worker, args=(args, mode), nprocs=args.world_size, join=False)
        start = time.monotonic()
        try:
            while not context.join(timeout=1):
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError(f"{mode} benchmark exceeded {args.timeout}s")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
    if args.mode == "both":
        reports = {mode: json.loads((args.output / f"{mode}.json").read_text())
                   for mode in ("unsharded", "sharded")}
        ratio = reports["sharded"]["mean_step_ms"] / reports["unsharded"]["mean_step_ms"]
        comparison = {"sharded_to_unsharded_time_ratio": ratio,
                      "slowdown_percent": (ratio - 1) * 100,
                      "reports": reports}
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(f"sharded/unsharded time ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
