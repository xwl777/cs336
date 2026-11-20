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
    d_scalar = torch.sum(o.float() * grad_o.float(), dim=-1)
    s = torch.einsum("... q d, ... k d -> ... q k", q.float(), k.float()) * scale
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[None, :, None] >= torch.arange(n_keys, device=q.device)[None, None, :]
        s = torch.where(mask, s, -1e6)

    p = torch.exp(s - l[..., :, None])
    d_v = torch.einsum("... q k, ... q d -> ... k d", p.to(grad_o.dtype), grad_o)
    d_p = torch.einsum("... q d, ... k d -> ... q k", grad_o.float(), v.float())
    d_s = p * (d_p - d_scalar[..., :, None])
    d_q = torch.einsum("... q k, ... k d -> ... q d", d_s.to(k.dtype), k) * scale
    d_k = torch.einsum("... q k, ... q d -> ... k d", d_s.to(q.dtype), q) * scale
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
            s = tl.dot(q, tl.trans(k), input_precision="tf32x3") * scale
            if is_causal:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                s = tl.where(causal_mask, s, s + -1.0e6)
            s = tl.where(key_offsets[None, :] < N_KEYS, s, -float("inf"))

            m_next = tl.maximum(m, tl.max(s, axis=1))
            p_tilde = tl.exp(s - m_next[:, None])
            alpha = tl.exp(m - m_next)
            l = alpha * l + tl.sum(p_tilde, axis=1)
            acc = tl.dot(p_tilde.to(V_block_ptr.type.element_ty), v, acc=acc * alpha[:, None], input_precision="tf32x3")
            m = m_next

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
            key_offsets += K_TILE_SIZE

        o = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, lse, boundary_check=(0,))


    @triton.jit
    def flash_bwd_preprocess_kernel(
        O_ptr, DO_ptr, Delta_ptr,
        N_QUERIES: tl.constexpr, D: tl.constexpr, ROW_TILE_SIZE: tl.constexpr,
    ):
        batch = tl.program_id(1)
        row_start = tl.program_id(0) * ROW_TILE_SIZE
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(row_start, 0),
            block_shape=(ROW_TILE_SIZE, D), order=(1, 0),
        )
        DO_block_ptr = tl.make_block_ptr(
            DO_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(row_start, 0),
            block_shape=(ROW_TILE_SIZE, D), order=(1, 0),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch * N_QUERIES,
            shape=(N_QUERIES,), strides=(1,), offsets=(row_start,),
            block_shape=(ROW_TILE_SIZE,), order=(0,),
        )
        o = tl.load(O_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(Delta_block_ptr, delta, boundary_check=(0,))


    @triton.jit
    def _flash_bwd_prob_and_ds(
        q, k, v, do, lse, delta, query_rows, key_rows,
        N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr,
        scale, is_causal: tl.constexpr,
    ):
        scores = tl.dot(q, tl.trans(k), input_precision="tf32x3") * scale
        if is_causal:
            scores = tl.where(query_rows[:, None] >= key_rows[None, :], scores, scores - 1.0e6)
        valid = (query_rows[:, None] < N_QUERIES) & (key_rows[None, :] < N_KEYS)
        scores = tl.where(valid, scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        dp = tl.dot(do, tl.trans(v), input_precision="tf32x3")
        ds = p * (dp - delta[:, None])
        return p, ds


    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, DO_ptr, L_ptr, Delta_ptr, DK_ptr, DV_ptr,
        N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr, D: tl.constexpr,
        scale, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        # Each key program owns complete dK/dV rows; no atomic accumulation.
        batch = tl.program_id(1)
        key_start = tl.program_id(0) * K_TILE_SIZE
        key_rows = key_start + tl.arange(0, K_TILE_SIZE)
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        DK_block_ptr = tl.make_block_ptr(
            DK_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        DV_block_ptr = tl.make_block_ptr(
            DV_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        dk = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        query_start = 0
        if is_causal:
            query_start = (key_start // Q_TILE_SIZE) * Q_TILE_SIZE
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(query_start, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        DO_block_ptr = tl.make_block_ptr(
            DO_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(query_start, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch * N_QUERIES,
            shape=(N_QUERIES,), strides=(1,), offsets=(query_start,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch * N_QUERIES,
            shape=(N_QUERIES,), strides=(1,), offsets=(query_start,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )

        for start in range(query_start, N_QUERIES, Q_TILE_SIZE):
            query_rows = start + tl.arange(0, Q_TILE_SIZE)
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
            do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
            lse = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")
            p, ds = _flash_bwd_prob_and_ds(
                q, k, v, do, lse, delta, query_rows, key_rows,
                N_QUERIES, N_KEYS, scale, is_causal,
            )
            dv = tl.dot(tl.trans(p.to(do.dtype)), do, acc=dv, input_precision="tf32x3")
            dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk, input_precision="tf32x3")
            Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
            DO_block_ptr = DO_block_ptr.advance((Q_TILE_SIZE, 0))
            L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
            Delta_block_ptr = Delta_block_ptr.advance((Q_TILE_SIZE,))

        tl.store(DK_block_ptr, (dk * scale).to(DK_block_ptr.type.element_ty), boundary_check=(0,))
        tl.store(DV_block_ptr, dv.to(DV_block_ptr.type.element_ty), boundary_check=(0,))


    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, DO_ptr, L_ptr, Delta_ptr, DQ_ptr,
        N_QUERIES: tl.constexpr, N_KEYS: tl.constexpr, D: tl.constexpr,
        scale, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        batch = tl.program_id(1)
        query_start = tl.program_id(0) * Q_TILE_SIZE
        query_rows = query_start + tl.arange(0, Q_TILE_SIZE)
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(query_start, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        DO_block_ptr = tl.make_block_ptr(
            DO_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(query_start, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        DQ_block_ptr = tl.make_block_ptr(
            DQ_ptr + batch * N_QUERIES * D,
            shape=(N_QUERIES, D), strides=(D, 1), offsets=(query_start, 0),
            block_shape=(Q_TILE_SIZE, D), order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch * N_QUERIES,
            shape=(N_QUERIES,), strides=(1,), offsets=(query_start,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch * N_QUERIES,
            shape=(N_QUERIES,), strides=(1,), offsets=(query_start,),
            block_shape=(Q_TILE_SIZE,), order=(0,),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch * N_KEYS * D,
            shape=(N_KEYS, D), strides=(D, 1), offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D), order=(1, 0),
        )
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        do = tl.load(DO_block_ptr, boundary_check=(0,), padding_option="zero")
        lse = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")
        dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        key_end = N_KEYS
        if is_causal:
            key_end = tl.minimum(N_KEYS, query_start + Q_TILE_SIZE)

        for start in range(0, key_end, K_TILE_SIZE):
            key_rows = start + tl.arange(0, K_TILE_SIZE)
            k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
            _, ds = _flash_bwd_prob_and_ds(
                q, k, v, do, lse, delta, query_rows, key_rows,
                N_QUERIES, N_KEYS, scale, is_causal,
            )
            dq = tl.dot(ds.to(k.dtype), k, acc=dq, input_precision="tf32x3")
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        tl.store(DQ_block_ptr, (dq * scale).to(DQ_block_ptr.type.element_ty), boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16
    BWD_Q_TILE_SIZE = 32
    BWD_K_TILE_SIZE = 64

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is required for FlashAttentionTriton.")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("FlashAttentionTriton expects CUDA tensors.")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Expected Q/K/V with shape (batch, sequence, embedding).")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q/K/V must have the same device and dtype.")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Triton attention supports float16, bfloat16 and float32 inputs.")
        if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Q/K/V batch and embedding dimensions must match; K/V shapes must match.")
        if q.shape[-1] < 16 or q.shape[-1] & (q.shape[-1] - 1) or min(q.shape[1], k.shape[1]) < 1:
            raise ValueError("Embedding dimension must be a power of two >= 16; sequence lengths must be positive.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        q_tile_size = FlashAttentionTriton.Q_TILE_SIZE
        k_tile_size = FlashAttentionTriton.K_TILE_SIZE
        tiles = (q_tile_size, k_tile_size, FlashAttentionTriton.BWD_Q_TILE_SIZE, FlashAttentionTriton.BWD_K_TILE_SIZE)
        if any(tile < 16 or tile & (tile - 1) for tile in tiles):
            raise ValueError("Attention tile sizes must be powers of two >= 16.")
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
        ctx.bwd_q_tile_size = FlashAttentionTriton.BWD_Q_TILE_SIZE
        ctx.bwd_k_tile_size = FlashAttentionTriton.BWD_K_TILE_SIZE
        return o

    @staticmethod
    def backward(ctx, grad_o: torch.Tensor):
        lse, q, k, v, o = ctx.saved_tensors
        grad_o = grad_o.contiguous()
        batch_size, n_queries, d = q.shape
        n_keys = k.shape[1]
        q_tile_size, k_tile_size = ctx.bwd_q_tile_size, ctx.bwd_k_tile_size
        delta = torch.empty_like(lse)
        d_q, d_k, d_v = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        scale = 1.0 / math.sqrt(d)
        # All launches use the current stream; D is ready before either consumer.
        with torch.cuda.device(q.device):
            flash_bwd_preprocess_kernel[(triton.cdiv(n_queries, q_tile_size), batch_size)](
                o, grad_o, delta, N_QUERIES=n_queries, D=d, ROW_TILE_SIZE=q_tile_size,
            )
            flash_bwd_dkdv_kernel[(triton.cdiv(n_keys, k_tile_size), batch_size)](
                q, k, v, grad_o, lse, delta, d_k, d_v,
                N_QUERIES=n_queries, N_KEYS=n_keys, D=d, scale=scale,
                Q_TILE_SIZE=q_tile_size, K_TILE_SIZE=k_tile_size, is_causal=ctx.is_causal,
            )
            flash_bwd_dq_kernel[(triton.cdiv(n_queries, q_tile_size), batch_size)](
                q, k, v, grad_o, lse, delta, d_q,
                N_QUERIES=n_queries, N_KEYS=n_keys, D=d, scale=scale,
                Q_TILE_SIZE=q_tile_size, K_TILE_SIZE=k_tile_size, is_causal=ctx.is_causal,
            )
        return (d_q, d_k, d_v, None)[:len(ctx.needs_input_grad)]
