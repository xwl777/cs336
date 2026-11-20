from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.multiprocessing as mp

from .adapters import get_sharded_optimizer
from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    # Since we've seeded, model states should be the same across ranks without having to broadcast.
    non_sharded_model = model_class().to(device)

    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(
        sharded_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()

        # batch size 32, 10 input features, 5 output features
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 10)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        non_sharded_optimizer.step()
        sharded_optimizer.step()

    # Check that the final model weights are the same regardless of if we're using
    # the sharded or non-sharded optimizer.
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )
    _cleanup_process_group()


def test_sharded_optimizer_param_groups():
    mp.spawn(_test_sharded_optimizer_param_groups, args=(2,), nprocs=2, join=True)


def _test_sharded_optimizer_param_groups(rank, world_size):
    import torch.distributed as dist

    _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        parameters = [torch.nn.Parameter(torch.tensor([float(i + 1)])) for i in range(3)]
        baseline = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        # Initially rank 1 owns no parameters: its local group must be empty.
        sharded = get_sharded_optimizer([{"params": parameters[:1], "lr": 0.01}], torch.optim.AdamW, lr=0.1)
        ordinary = torch.optim.AdamW([{"params": baseline[:1], "lr": 0.01}], lr=0.1)
        for optimizer, values in [(sharded, parameters), (ordinary, baseline)]:
            optimizer.add_param_group({"params": values[1:], "lr": 0.2})

        for iteration in range(3):
            # Emulate a scheduler editing the public groups.
            sharded.param_groups[0]["lr"] = ordinary.param_groups[0]["lr"] = 0.03
            for optimizer, values in [(sharded, parameters), (ordinary, baseline)]:
                def closure():
                    optimizer.zero_grad()
                    loss = sum(p.square().sum() for p in values)
                    loss.backward()
                    return loss
                result = optimizer.step(closure=closure)
                assert result is not None
            for parameter, expected in zip(parameters, baseline):
                torch.testing.assert_close(parameter, expected)
                replica = parameter.detach().clone()
                dist.broadcast(replica, src=0)
                torch.testing.assert_close(parameter, replica)
            assert set(sharded.state) == {p for i, p in enumerate(parameters) if i % world_size == rank}
            # Local checkpoint restore must keep the wrapped state connected.
            sharded.load_state_dict(deepcopy(sharded.state_dict()))
    finally:
        _cleanup_process_group()
