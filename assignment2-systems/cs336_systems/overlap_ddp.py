"""Overlap DDP: all-reduce each gradient asynchronously the moment it is ready.

Gradients become available incrementally during backward (from the loss toward
the input), so we register a post-accumulate hook on every trainable parameter
that launches a non-blocking all-reduce. Communication then overlaps with the
rest of the backward pass; finish_gradient_synchronization waits for the
in-flight operations before the optimizer step.

With gradient accumulation, wrap every backward except the last one of a step
in no_sync() so gradients are accumulated locally and communicated once,
matching the once-per-step semantics of NaiveDDP/FlatDDP.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn


class OverlapDDP(nn.Module):
    """Wrap a model using the default, already-initialized process group.

    The caller assigns each process its device, shards the batch equally, and
    constructs the optimizer after wrapping. All ranks run identical backward
    passes, so hooks fire in the same order and the per-parameter all-reduces
    pair up correctly across ranks.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group before creating OverlapDDP.")
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles: list[tuple[torch.Tensor, dist.Work]] = []
        self._sync_enabled = True

        # Include frozen parameters and buffers so all replicas start identically.
        # Module.parameters() deduplicates tied weights automatically.
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)

        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue
            parameter.register_post_accumulate_grad_hook(self._allreduce_gradient)

    @contextmanager
    def no_sync(self):
        """Skip gradient communication for backwards inside this context.

        Use around all but the last micro-batch of a gradient accumulation
        cycle; hooks still fire but only accumulate gradients locally.
        """
        previous = self._sync_enabled
        self._sync_enabled = False
        try:
            yield
        finally:
            self._sync_enabled = previous

    def _allreduce_gradient(self, parameter: torch.Tensor) -> None:
        # Fires once param.grad holds the full local gradient for this backward.
        if not self._sync_enabled:
            return

        # Launch now, but defer waiting and averaging until backward finishes.
        # Do not modify this gradient again before finish_gradient_synchronization.
        handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((parameter, handle))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Wait and average every gradient before the optimizer step.

        Call before another backward, zero_grad, or optimizer step can modify
        the gradients that are still being reduced.
        """
        for parameter, handle in self.handles:
            handle.wait()
            parameter.grad.div_(self.world_size)
        self.handles.clear()
