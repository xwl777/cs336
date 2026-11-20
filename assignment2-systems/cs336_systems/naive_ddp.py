"""Naive DDP: synchronize each parameter gradient after backward completes."""

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    """Wrap a model using the default, already-initialized process group.

    The caller assigns each process its device, shards the batch equally, and
    constructs the optimizer after wrapping. Call finish_gradient_synchronization
    once after backward and before each optimizer step. All ranks must use the
    same parameters in backward and call synchronization in the same order.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group before creating NaiveDDP.")
        self.module = module
        self.world_size = dist.get_world_size()

        # Include frozen parameters and buffers so all replicas start identically.
        # Module.parameters() deduplicates tied weights automatically.
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """Average each gradient in place; no communication/backward overlap."""
        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=False)
            parameter.grad.div_(self.world_size)
