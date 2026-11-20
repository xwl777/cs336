"""Shard optimizer state by assigning each parameter to one updating rank.

Parameters and synchronized gradients remain replicated. Only the owner keeps
optimizer state and updates a parameter; broadcasts replicate the new values.
All ranks must supply identical parameter groups in identical order.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        if not dist.is_initialized():
            raise RuntimeError("Initialize the distributed process group before creating ShardedOptimizer.")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._owners: dict[torch.Tensor, int] = {}
        self._local_optimizer: Optimizer | None = None
        # Optimizer.__init__ calls our add_param_group for each input group.
        super().__init__(params, kwargs)
        local_groups = [self._local_group(group) for group in self.param_groups]
        # Retain empty groups so ranks with no owned parameters can participate.
        self._local_optimizer = optimizer_cls(local_groups, **kwargs)
        self.defaults = self._local_optimizer.defaults.copy()
        for group, local_group in zip(self.param_groups, self._local_optimizer.param_groups):
            for key, value in local_group.items():
                if key != "params":
                    group[key] = value
        # Expose only this rank's optimizer state through the wrapper.
        self.state = self._local_optimizer.state

    def _local_group(self, group: dict[str, Any]) -> dict[str, Any]:
        return {
            **group,
            "params": [p for p in group["params"] if self._owners[p] == self.rank],
        }

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        new_parameters = []
        for parameter in group["params"]:
            if parameter not in self._owners:
                self._owners[parameter] = len(self._owners) % self.world_size
                new_parameters.append(parameter)
        if self._local_optimizer is not None:
            try:
                self._local_optimizer.add_param_group(self._local_group(group))
            except Exception:
                self.param_groups.pop()
                for parameter in new_parameters:
                    del self._owners[parameter]
                raise

    def step(self, closure=None, **kwargs: Any):
        # Schedulers/users modify public groups; forward their current settings.
        for group, local_group in zip(self.param_groups, self._local_optimizer.param_groups):
            for key, value in group.items():
                if key != "params":
                    local_group[key] = value
        result = self._local_optimizer.step(closure=closure, **kwargs)
        with torch.no_grad():
            for parameter, owner in self._owners.items():
                dist.broadcast(parameter, src=owner)
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # The inherited loader uses the full public parameter-group indexing.
        super().load_state_dict(state_dict)
        self._local_optimizer.state = self.state
