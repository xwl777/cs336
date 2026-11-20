"""Flat DDP: concatenate all gradients, issue a single all-reduce, then scatter back.

Same calling contract as NaiveDDP: assign a device per process, shard batches,
construct the optimizer after wrapping, and call
finish_gradient_synchronization once after backward and before each optimizer
step.
"""

import torch
import torch.distributed as dist
from torch import nn


class FlatDDP(nn.Module):
    """Batch every parameter gradient into one buffer and synchronize it with a
    single all-reduce call, trading many small collectives for one large one."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group before creating FlatDDP.")
        self.module = module
        self.world_size = dist.get_world_size()

        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """All-reduce the flattened gradients once; no communication/backward overlap."""
        parameters = [p for p in self.module.parameters() if p.grad is not None]
        if not parameters:
            return

        grads = [p.grad for p in parameters]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
        flat.div_(self.world_size)

        # Unflatten returns views into `flat`, so assigning them keeps every
        # gradient in one contiguous, already-synchronized buffer.
        for parameter, view in zip(parameters, torch._utils._unflatten_dense_tensors(flat, parameters)):
            parameter.grad = view
