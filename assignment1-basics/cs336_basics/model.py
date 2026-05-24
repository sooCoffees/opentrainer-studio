from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

from cs336_basics.implementation import rmsnorm, rope, scaled_dot_product_attention, swiglu


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1024
    rope_theta: float = 10000.0
    init_std: float = 0.02
    attention_backend: str = "sdpa"

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, init_std: float = 0.02):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, init_std)
        self.w2 = Linear(d_ff, d_model, init_std)
        self.w3 = Linear(d_model, d_ff, init_std)

    def forward(self, x: Tensor) -> Tensor:
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        rope_theta: float,
        init_std: float = 0.02,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attention_backend not in {"sdpa", "manual"}:
            raise ValueError("attention_backend must be 'sdpa' or 'manual'")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.attention_backend = attention_backend
        self.q_proj = Linear(d_model, d_model, init_std)
        self.k_proj = Linear(d_model, d_model, init_std)
        self.v_proj = Linear(d_model, d_model, init_std)
        self.output_proj = Linear(d_model, d_model, init_std)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context length {self.context_length}")
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        q = rope(self.head_dim, self.rope_theta, self.context_length, q, token_positions)
        k = rope(self.head_dim, self.rope_theta, self.context_length, k, token_positions)
        if self.attention_backend == "sdpa":
            x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = self.causal_mask[:seq_len, :seq_len]
            x = scaled_dot_product_attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.output_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(
            config.d_model,
            config.num_heads,
            config.context_length,
            config.rope_theta,
            config.init_std,
            config.attention_backend,
        )
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff, config.init_std)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model, config.init_std)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_final = RMSNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, config.init_std)

    def forward(self, in_indices: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        if in_indices.shape[-1] > self.config.context_length:
            raise ValueError("input sequence exceeds configured context_length")
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_final(x))
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        end_token_id: int | None = None,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        out = list(token_ids)
        for _ in range(max_new_tokens):
            context = out[-self.config.context_length :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(x)
            assert isinstance(logits, Tensor)
            next_logits = logits[0, -1] / max(temperature, 1e-8)
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.numel()))
                next_logits = next_logits.masked_fill(next_logits < values[-1], -torch.inf)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            out.append(next_id)
            if end_token_id is not None and next_id == end_token_id:
                break
        return out
