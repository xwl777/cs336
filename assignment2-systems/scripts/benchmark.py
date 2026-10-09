import argparse
import timeit
import statistics
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

import torch
from cs336_basics.model import Transformer
from cs336_basics.optimizer import AdamW
import cs336_basics.loss


def synchronize(device:torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def build_model(args, device):
    return Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        device=device,
        dtype=torch.float32
    )

def get_bacth(args, device):
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device=device
    )

    y = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            device=device
        )
    return x, y

def run_step(model:torch.nn.Module, 
             mode:str, 
             optimizer:torch.optim.Optimizer,
             x:torch.Tensor, 
             y:torch.Tensor,
             amp_context
             ):
    if mode == "forward":
        with nvtx.range("forward"):
            with torch.no_grad():
                with amp_context:
                    logits = model(x)
        return logits

    elif mode in {"forward_backward", "train"}:
        with nvtx.range("forward"):
            with amp_context:
                logits=model(x)
        with nvtx.range("loss compute"):
            with amp_context:
                loss = cs336_basics.loss.cross_entropy(logits=logits, targets=y)
        with nvtx.range("backward"):
            loss.backward()
        if mode == "train":
            with nvtx.range("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

        return loss
    else:
        raise ValueError("mode is not defined")

def benchmark(args):
    device = torch.device(args.device)
    model = build_model(args=args,device=device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = get_bacth(args, device=device)
    amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.mix_precision else nullcontext()

    for _ in range(args.warmup_steps):
        optimizer.zero_grad()
        run_step(model, args.mode, optimizer, x, y, amp_context)
        synchronize(device)


    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

        optimizer.zero_grad(set_to_none=True)
        with nvtx.range("memory_profile"):
            run_step(model, args.mode, optimizer, x, y, amp_context)
            synchronize(device)

        torch.cuda.memory._dump_snapshot(args.memory_profile_path)
        torch.cuda.memory._record_memory_history(enabled=None)

        print(f"memory snapshot written to {args.memory_profile_path}")
        return

    times = []
    with nvtx.range("benchmark"):
        for _ in range(args.measurement_steps):
            optimizer.zero_grad(set_to_none=True)
            start = timeit.default_timer()
            run_step(model, args.mode, optimizer, x, y, amp_context)
            synchronize(device)
            end = timeit.default_timer()
            times.append(end - start)
        

    mean = statistics.mean(times)
    std = statistics.stdev(times)

    print(f"mode: {args.mode}")
    print(f"device: {device}")
    print(f"mean: {mean:.6f} s")
    print(f"std:  {std:.6f} s")
    print(f"times: {[round(t, 6) for t in times]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",choices=["forward", "forward_backward", "train"],
      required=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)

    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measurement_steps", type=int, default=10)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--mix_precision", action="store_true")
    parser.add_argument("--memory_profile", action="store_true")
    parser.add_argument("--memory_profile_path", type=str, default="memory_snapshot.pickle")
    args = parser.parse_args()

    benchmark(args)