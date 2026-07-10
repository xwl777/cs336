import torch.nn as nn
import torch
import math
from einops import einsum, rearrange, reduce


def softmax(x:torch.Tensor, dim:int):
    x_max = x.max(dim = dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor | None = None):
    d_k = q.shape[-1]
    atten = einsum(q, k, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k") / math.sqrt(d_k)
    if mask is not None:
        atten = torch.where(mask, atten, float("-inf"))
    
    score = softmax(atten, dim = -1)

    return einsum(score, v, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(w)

    def forward(self, x:torch.Tensor):
        return einsum(x, self.weight, "... in_dim, out_dim in_dim -> ... out_dim")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        pass
        super().__init__()
        std = 1
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(w, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(w)

    def forward(self, token_ids:torch.Tensor):
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        w = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(w)

    def forward(self, x:torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32) #保证平方时数值稳定

        result = torch.rsqrt(reduce(x**2, "... d -> ... 1", "mean") + self.eps) * x * self.weight
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int | None = None, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        if not d_ff:
            d_ff = round(d_model /24) * 64 # 24来源于8 / 3 / 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        

    def _silu_activation(self, x:torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor):
        return self.w2(self._silu_activation(self.w1(x)) * self.w3(x))
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len:int, device:torch.device | None = None):
        super().__init__()
        freqs = torch.arange(0, d_k, 2, device=device) / d_k
        freqs = 1 / theta ** freqs
        pos = torch.arange(0, max_seq_len, device=device)
        angles = einsum(pos, freqs, "seq_len, d -> seq_len d")

        cos = angles.cos()
        sin = angles.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)



    def forward(self, x:torch.Tensor, token_positions:torch.Tensor)->torch.Tensor:  
        d_k = x.shape[-1]
        cos_star = self.cos[token_positions]
        sin_star = self.sin[token_positions]
        even_half = x[...,0::2]
        odd_half = x[...,1::2]

        x1 = even_half * cos_star - odd_half * sin_star
        x2 = even_half * sin_star + odd_half * cos_star

        return rearrange(torch.stack([x1, x2], dim = -1), "... d two -> ... (d two)")
    
class causal_multihead_self_attention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor, rope:RotaryPositionalEmbedding | None = None, token_positions:torch.Tensor | None = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        seq_len = x.shape[-2]
        device = x.device

        q = rearrange(q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)

        if rope is not None:
            if token_positions is None:
                pos = torch.arange(seq_len,device=device)
            else:
                pos = token_positions

            q = rope(q, pos)
            k = rope(k, pos)

        

        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

        latent = scaled_dot_product_attention(q, k, v, mask=mask)
        latent = rearrange(latent, "... num_heads seq_len h_head -> ... seq_len (num_heads h_head)")

        return self.output_proj(latent)
    

class Transformer_block(nn.Module):
    def __init__(self, d_model:int,
                  num_heads:int, 
                  d_ff:int, 
                  rope:RotaryPositionalEmbedding | None = None,
                  device:torch.device | None = None, 
                  dtype:torch.dtype | None = None):
        super().__init__()
        self.rope = rope
        self.attn = causal_multihead_self_attention(d_model, num_heads, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), self.rope, token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, d_model:int,
                 num_heads:int,
                 d_ff:int,
                 theta:float,
                 context_length:int,
                 vocab_size:int,
                 num_layers:int,
                 device : torch.device | None = None,
                 dtype : torch.dtype | None = None):
        super().__init__()
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList([Transformer_block(d_model, num_heads, d_ff, rope, device=device, dtype = dtype) 
                                     for _ in range(num_layers)])
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor | None = None):
        seq_len = x.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max context length ({self.context_length})")
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x, token_positions = token_positions)

        return self.lm_head(self.ln_final(x))