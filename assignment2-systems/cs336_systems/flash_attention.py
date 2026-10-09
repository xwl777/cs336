from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is only needed for CUDA tests.
    triton = None
    tl = None


@torch.compile(fullgraph=True)
def _flash_backward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    l: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    # D is a row-wise vector; saved L lets us reconstruct P without softmax.
    d_scalar = torch.sum(o * grad_o, dim=-1)
    s = torch.einsum("... q d, ... k d -> ... q k", q, k) * scale
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[None, :, None] >= torch.arange(n_keys, device=q.device)[None, None, :]
        s = torch.where(mask, s, -1e6)

    p = torch.exp(s - l[..., :, None])
    d_v = torch.einsum("... q k, ... q d -> ... k d", p, grad_o)
    d_p = torch.einsum("... q d, ... k d -> ... q k", grad_o, v)
    d_s = p * (d_p - d_scalar[..., :, None])
    d_q = torch.einsum("... q k, ... k d -> ... q d", d_s, k) * scale
    d_k = torch.einsum("... q k, ... q d -> ... k d", d_s, q) * scale
    return d_q, d_k, d_v


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        d = q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        q_tile_size = 16
        k_tile_size = 16

        outputs = []
        lse_tiles = []
        for q_start in range(0, q.shape[-2], q_tile_size):
            q_tile = q[..., q_start : q_start + q_tile_size, :]
            o_tile = torch.zeros_like(q_tile)
            m = torch.full(q_tile.shape[:-1], -torch.inf, device=q.device, dtype=torch.float32)
            l = torch.zeros(q_tile.shape[:-1], device=q.device, dtype=torch.float32)

            query_positions = torch.arange(q_start, q_start + q_tile.shape[-2], device=q.device)
            for k_start in range(0, k.shape[-2], k_tile_size):
                k_tile = k[..., k_start : k_start + k_tile_size, :]
                v_tile = v[..., k_start : k_start + k_tile_size, :]
                s = torch.einsum("... q d, ... k d -> ... q k", q_tile, k_tile) * scale
                if is_causal:
                    key_positions = torch.arange(k_start, k_start + k_tile.shape[-2], device=q.device)
                    mask = query_positions[..., None] >= key_positions[None, :]
                    s = torch.where(mask, s, -1e6)

                m_next = torch.maximum(m, torch.max(s, dim=-1).values)
                p_tilde = torch.exp(s - m_next[..., None])
                alpha = torch.exp(m - m_next)
                l = alpha * l + torch.sum(p_tilde, dim=-1)
                o_tile = alpha[..., None] * o_tile + torch.einsum("... q k, ... k d -> ... q d", p_tilde.to(v.dtype), v_tile)
                m = m_next

            outputs.append((o_tile / l[..., None]).to(q.dtype))
            lse_tiles.append(m + torch.log(l))

        o = torch.cat(outputs, dim=-2)
        lse = torch.cat(lse_tiles, dim=-1)
        ctx.save_for_backward(lse, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        lse, q, k, v, o = ctx.saved_tensors
        d_q, d_k, d_v = _flash_backward_pytorch(q, k, v, o, grad_o, lse, ctx.is_causal)
        return d_q, d_k, d_v, None


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        N_QUERIES,
        N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_offsets = tl.arange(0, K_TILE_SIZE)

        for _ in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            s = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                s = tl.where(causal_mask, s, s + -1.0e6)

            m_next = tl.maximum(m, tl.max(s, axis=1))
            p_tilde = tl.exp(s - m_next[:, None])
            alpha = tl.exp(m - m_next)
            l = alpha * l + tl.sum(p_tilde, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p_tilde.to(V_block_ptr.type.element_ty), v)
            m = m_next

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
            key_offsets += K_TILE_SIZE

        o = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, lse, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is required for FlashAttentionTriton.")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("FlashAttentionTriton expects CUDA tensors.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        q_tile_size = 16
        k_tile_size = 16
        o = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(n_queries, q_tile_size), batch_size)
        flash_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            lse.stride(0),
            lse.stride(1),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(lse, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        lse, q, k, v, o = ctx.saved_tensors
        d_q, d_k, d_v = _flash_backward_pytorch(q, k, v, o, grad_o, lse, ctx.is_causal)
        return d_q, d_k, d_v, None
