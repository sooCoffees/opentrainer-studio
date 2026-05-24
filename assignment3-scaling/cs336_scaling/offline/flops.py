from __future__ import annotations

from cs336_scaling.offline.configs import ModelSpec


def estimate_transformer_params(model: ModelSpec) -> int:
    embeddings = model.vocab_size * model.d_model
    lm_head = 0
    attention = model.num_layers * 4 * model.d_model * model.d_model
    ffn = model.num_layers * (
        2 * model.d_model * model.d_ff + model.d_ff * model.d_model
    )
    norms = model.num_layers * 2 * model.d_model + model.d_model
    return embeddings + lm_head + attention + ffn + norms


def estimate_training_flops(params: int, train_tokens: int) -> float:
    return 6.0 * params * train_tokens


def train_tokens_for_compute(compute_budget: float, params: int) -> int:
    return max(1, int(compute_budget / (6.0 * params)))


def compute_optimal_tokens(params: int, token_multiplier: float = 20.0) -> int:
    return int(token_multiplier * params)


def compute_optimal_params(
    compute_budget: float, token_multiplier: float = 20.0
) -> int:
    return int((compute_budget / (6.0 * token_multiplier)) ** 0.5)
