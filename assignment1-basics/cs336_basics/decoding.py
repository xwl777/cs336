import torch

from cs336_basics.model import *
from cs336_basics.bpe import *
def decode(
        model:torch.nn.Module,
        tokenizer:Tokenizer,
        prompt:str,
        context_length:int,
        max_new_tokens:int,
        temperature:float,
        top_p:float,
        end_of_text:str
) -> str:
    end_id = tokenizer.encode(end_of_text)[0]
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-context_length:] if context_length < len(prompt) else input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.Tensor(input_ids, device=device)
            logits = model(x)[0, -1, :]
            logits /= temperature
            logits -= logits.max(dim = -1)
            exp = torch.exp(logits)
            exp_sum = exp.sum(dim = -1)
            probs = exp / exp_sum
            probs_sorted, idx_sorted = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(probs_sorted)
            cutoff_idx = torch.searchsorted(cumsum, top_p)

            trimmed_probs = probs_sorted[:cutoff_idx]
            trimmed_idxs = idx_sorted[:cutoff_idx]
            trimmed_probs /= trimmed_probs.sum(dim = -1)

            next_token = trimmed_idxs[torch.multinomial(trimmed_probs, 1).item()]
            input_ids.append(next_token.item())
            if next_token.item() == end_id:
                break
    return tokenizer.decode(input_ids)
