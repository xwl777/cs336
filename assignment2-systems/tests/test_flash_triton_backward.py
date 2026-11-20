import math

import pytest
import torch

from cs336_systems.flash_attention import FlashAttentionTriton


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires a CUDA GPU")
@pytest.mark.parametrize("shape", [(1, 37, 51, 16), (2, 96, 80, 64), (1, 128, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_backward_triton_tiled(shape, dtype, is_causal, monkeypatch):
    pytest.importorskip("triton")
    batch, nq, nk, d = shape
    monkeypatch.setattr(FlashAttentionTriton, "Q_TILE_SIZE", 32)
    monkeypatch.setattr(FlashAttentionTriton, "K_TILE_SIZE", 64)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(42)
    q = torch.randn((batch, nq, d), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((batch, nk, d), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # Exercise the contiguous conversion for a strided upstream gradient.
    grad_o = torch.randn((batch, d, nq), device="cuda", dtype=dtype).transpose(1, 2)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    if is_causal:
        mask = torch.arange(nq, device="cuda")[:, None] >= torch.arange(nk, device="cuda")[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    reference = torch.softmax(scores, dim=-1).to(dtype) @ v
    expected = torch.autograd.grad(reference, (q, k, v), grad_o)
    actual_output = FlashAttentionTriton.apply(q, k, v, True) if is_causal else FlashAttentionTriton.apply(q, k, v)
    actual = torch.autograd.grad(actual_output, (q, k, v), grad_o)
    tolerance = 5e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(actual_output, reference, atol=tolerance, rtol=tolerance)
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
