from collections.abc import Iterable
import torch
def gradient_clip(params:Iterable[torch.nn.Parameter], max_norm:float):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    
    norm = torch.norm(torch.stack([torch.norm(g.detach(), p = 2) for g in grads]), p = 2)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads:
            g.detach().mul_(scale)

    return norm

    