from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator

import numpy as np
import regex as re
import torch
from torch import Tensor


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def linear(x: Tensor, weight: Tensor) -> Tensor:
    return x @ weight.transpose(-1, -2)


def embedding(token_ids: Tensor, weight: Tensor) -> Tensor:
    return weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def swiglu(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return linear(silu(linear(x, w1)) * linear(x, w3), w2)


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    dtype = x.dtype
    x_float = x.float()
    scale = torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps)
    return (x_float * scale * weight.float()).to(dtype)


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    maxes = torch.max(inputs, dim=-1, keepdim=True).values
    logsumexp = maxes.squeeze(-1) + torch.log(torch.sum(torch.exp(inputs - maxes), dim=-1))
    return torch.mean(logsumexp - inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return softmax(scores, dim=-1) @ V


def rope(d_k: int, theta: float, max_seq_len: int, x: Tensor, token_positions: Tensor) -> Tensor:
    del max_seq_len
    half = d_k // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) * 2 / d_k))
    angles = token_positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * freqs
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(dtype=x.dtype)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
    return x * cos + rotated * sin


def multihead_self_attention(
    x: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    max_seq_len: int | None = None,
    theta: float | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    *batch_dims, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    q = linear(x, q_proj_weight).reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    k = linear(x, k_proj_weight).reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    v = linear(x, v_proj_weight).reshape(*batch_dims, seq_len, num_heads, head_dim).transpose(-3, -2)
    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).expand(*batch_dims, seq_len)
        q = rope(head_dim, theta, max_seq_len or seq_len, q, token_positions)
        k = rope(head_dim, theta, max_seq_len or seq_len, k, token_positions)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
    attended = scaled_dot_product_attention(q, k, v, causal_mask)
    attended = attended.transpose(-3, -2).reshape(*batch_dims, seq_len, d_model)
    return linear(attended, o_proj_weight)


def transformer_block(
    x: Tensor,
    weights: dict[str, Tensor],
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
) -> Tensor:
    del d_ff
    h = x + multihead_self_attention(
        rmsnorm(x, weights["ln1.weight"]),
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        max_seq_len=max_seq_len,
        theta=theta,
    )
    return h + swiglu(
        rmsnorm(h, weights["ln2.weight"]), weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"]
    )


def transformer_lm(
    in_indices: Tensor,
    weights: dict[str, Tensor],
    num_layers: int,
    num_heads: int,
    d_ff: int,
    context_length: int,
    rope_theta: float,
) -> Tensor:
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {k.removeprefix(prefix): v for k, v in weights.items() if k.startswith(prefix)}
        x = transformer_block(x, layer_weights, num_heads, d_ff, context_length, rope_theta)
    x = rmsnorm(x, weights["ln_final.weight"])
    return linear(x, weights["lm_head.weight"])


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts])
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    total = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if total > max_l2_norm:
        scale = max_l2_norm / (total + 1e-6)
        for p in params:
            p.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
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
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.addcdiv_(exp_avg, torch.sqrt(exp_avg_sq).add_(eps), value=-step_size)
                p.mul_(1 - lr * weight_decay)
        return loss


def cosine_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_token_ids = {token: self.byte_to_id[token.encode("utf-8")] for token in self.special_tokens}
        self.pattern = re.compile(GPT2_PRETOKEN_PATTERN)
        self._cache: dict[bytes, list[int]] = {}
        self._special_re = (
            re.compile("|".join(re.escape(token) for token in self.special_tokens)) if self.special_tokens else None
        )

    def _encode_bytes(self, data: bytes) -> list[int]:
        if data in self._cache:
            return self._cache[data][:]
        parts = tuple(bytes([b]) for b in data)
        if not parts:
            return []
        while len(parts) > 1:
            pairs = [
                (self.merges[(parts[i], parts[i + 1])], i)
                for i in range(len(parts) - 1)
                if (parts[i], parts[i + 1]) in self.merges
            ]
            if not pairs:
                break
            _, merge_idx = min(pairs)
            merged = parts[merge_idx] + parts[merge_idx + 1]
            parts = parts[:merge_idx] + (merged,) + parts[merge_idx + 2 :]
        ids = [self.byte_to_id[p] for p in parts]
        self._cache[data] = ids
        return ids[:]

    def _encode_non_special(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self.pattern.finditer(text):
            ids.extend(self._encode_bytes(match.group(0).encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._special_re is None:
            return self._encode_non_special(text)
        ids: list[int] = []
        start = 0
        for match in self._special_re.finditer(text):
            ids.extend(self._encode_non_special(text[start : match.start()]))
            ids.append(self.special_token_ids[match.group(0)])
            start = match.end()
        ids.extend(self._encode_non_special(text[start:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    if special_tokens:
        special_re = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
        chunks = special_re.split(text)
    else:
        chunks = [text]

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    pattern = re.compile(GPT2_PRETOKEN_PATTERN)
    for chunk in chunks:
        for match in pattern.finditer(chunk):
            word_counts[tuple(bytes([b]) for b in match.group(0).encode("utf-8"))] += 1

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += count
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        new_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_counts.items():
            merged_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    merged_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            new_counts[tuple(merged_word)] += count
        word_counts = new_counts
    return vocab, merges
