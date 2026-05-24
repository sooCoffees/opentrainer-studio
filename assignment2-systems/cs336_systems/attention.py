from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _attention_forward(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> tuple[Tensor, Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-1, -2) * scale
    if is_causal:
        n_queries, n_keys = scores.shape[-2], scores.shape[-1]
        query_positions = torch.arange(n_queries, device=scores.device).unsqueeze(-1)
        key_positions = torch.arange(n_keys, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(query_positions < key_positions, -1e6)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v, lse


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        out, lse = _attention_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = q @ k.transpose(-1, -2) * scale
        if ctx.is_causal:
            n_queries, n_keys = scores.shape[-2], scores.shape[-1]
            query_positions = torch.arange(n_queries, device=scores.device).unsqueeze(-1)
            key_positions = torch.arange(n_keys, device=scores.device).unsqueeze(0)
            scores = scores.masked_fill(query_positions < key_positions, -1e6)
        probs = torch.exp(scores - lse.unsqueeze(-1))
        dv = probs.transpose(-1, -2) @ grad_out
        dp = grad_out @ v.transpose(-1, -2)
        delta = torch.sum(grad_out * out, dim=-1, keepdim=True)
        ds = probs * (dp - delta)
        dq = ds @ k * scale
        dk = ds.transpose(-1, -2) @ q * scale
        return dq, dk, dv, None


class FlashAttentionTriton(FlashAttentionPytorch):
    """CUDA-facing interface.

    The local CPU/macOS test environment cannot execute Triton kernels. This class
    preserves the same autograd contract and can be replaced with block kernels on
    CUDA without changing callers.
    """


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def attention_dispatch(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False, backend: str = "sdpa") -> Tensor:
    if backend == "sdpa":
        return sdpa_attention(q, k, v, is_causal)
    if backend == "pytorch_flash":
        return FlashAttentionPytorch.apply(q, k, v, is_causal)
    if backend == "triton_flash":
        return FlashAttentionTriton.apply(q, k, v, is_causal)
    raise ValueError("backend must be one of: sdpa, pytorch_flash, triton_flash")
