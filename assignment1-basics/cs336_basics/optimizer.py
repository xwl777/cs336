import torch
import math
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,
                 lr:float = 1e-3, 
                 betas:tuple[float, float] = (0.9, 0.95), 
                 eps:float = 1e-8, 
                 weight_decay:float = 0.1):
        defaults = {"lr":lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure:Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad
                alpha = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                p -= alpha * m / (torch.sqrt(v) + eps) + lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss