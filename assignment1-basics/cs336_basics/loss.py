import torch
from einops import reduce, rearrange

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    m = logits.max(dim = -1, keepdim=True).values
    exp = reduce(torch.exp(logits - m), "... vocab_size -> ...", "sum")
    m = m.squeeze(-1)
    loss = -logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) + m + torch.log(exp)
    loss = reduce(loss, "... -> ()", "mean")
    return loss

