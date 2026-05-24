from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingSpec:
    name: str
    model: ModelSpec
    train_tokens: int
    batch_tokens: int
    learning_rate: float
    min_learning_rate: float
    warmup_tokens: int
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    attention_backend: str = "sdpa"
    amp_dtype: str = "auto"
    compile: bool = True

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload


def round_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple) * multiple))


def make_model_spec(
    target_params: int,
    *,
    vocab_size: int = 10000,
    context_length: int = 512,
    name_prefix: str = "scale",
) -> ModelSpec:
    d_model = round_multiple(int(384 * (target_params / 10_000_000) ** 0.25), 64)
    num_heads = max(1, d_model // 64)
    embedding_params = vocab_size * d_model
    per_layer_params = 12 * d_model * d_model
    num_layers = max(
        2,
        round_multiple(
            int((target_params - embedding_params) / max(1, per_layer_params)), 2
        ),
    )
    d_ff = round_multiple(int((8 / 3) * d_model), 64)
    name = f"{name_prefix}_{target_params // 1_000_000}m"
    return ModelSpec(
        name=name,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )


def make_training_spec(
    model: ModelSpec,
    train_tokens: int,
    *,
    batch_tokens: int = 131072,
    base_lr: float = 6e-4,
) -> TrainingSpec:
    scale = (model.d_model / 384) ** -0.5
    lr = base_lr * scale
    return TrainingSpec(
        name=f"{model.name}_{train_tokens // 1_000_000}mtok",
        model=model,
        train_tokens=train_tokens,
        batch_tokens=batch_tokens,
        learning_rate=lr,
        min_learning_rate=lr * 0.1,
        warmup_tokens=max(batch_tokens, train_tokens // 100),
    )


def make_isoflops_grid(
    *,
    parameter_targets: list[int],
    compute_budgets: list[float],
    vocab_size: int = 10000,
    context_length: int = 512,
) -> list[TrainingSpec]:
    from cs336_scaling.offline.flops import train_tokens_for_compute

    specs: list[TrainingSpec] = []
    for compute_budget in compute_budgets:
        for target_params in parameter_targets:
            model = make_model_spec(
                target_params, vocab_size=vocab_size, context_length=context_length
            )
            train_tokens = train_tokens_for_compute(compute_budget, target_params)
            specs.append(make_training_spec(model, train_tokens))
    return specs
