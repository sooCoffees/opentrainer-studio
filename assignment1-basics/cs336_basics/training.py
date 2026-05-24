from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.implementation import AdamW, cosine_lr, get_batch
from cs336_basics.io import load_json, read_token_file, save_json
from cs336_basics.model import TransformerConfig, TransformerLM


@dataclass
class TrainConfig:
    train_bin: str
    valid_bin: str
    out_dir: str = "runs/debug"
    vocab_size: int = 1000
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 768
    rope_theta: float = 10000.0
    attention_backend: str = "sdpa"
    batch_size: int = 16
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    seed: int = 1337
    device: str = "auto"
    compile: bool = False
    amp_dtype: str | None = "auto"
    optimizer: str = "adamw"

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> TrainConfig:
        return cls(**load_json(path))

    def to_json(self, path: str | os.PathLike) -> None:
        save_json(asdict(self), path)

    def model_config(self) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            attention_backend=self.attention_backend,
        )


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: str, amp_dtype: str | None) -> torch.dtype | None:
    if amp_dtype is None or amp_dtype == "none" or device == "cpu":
        return None
    if amp_dtype == "auto":
        return torch.bfloat16 if device in {"cuda", "mps"} else None
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be one of: auto, bf16, fp16, none")


def build_optimizer(model: TransformerLM, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "torch_adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "fused_adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=torch.cuda.is_available(),
        )
    raise ValueError("optimizer must be one of: adamw, torch_adamw, fused_adamw")


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    config: TrainConfig,
    device: str,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split, data in (("train", train_data), ("valid", valid_data)):
        losses = []
        for _ in range(config.eval_iters):
            x, y = get_batch(data, config.batch_size, config.context_length, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save_training_checkpoint(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    iteration: int,
    best_valid_loss: float,
) -> None:
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "best_valid_loss": best_valid_loss,
        "model_config": model.config.to_dict(),
        "train_config": asdict(config),
    }
    torch.save(payload, out_dir / "checkpoint.pt")


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(out_dir / "config.json")

    train_data = read_token_file(config.train_bin, config.vocab_size)
    valid_data = read_token_file(config.valid_bin, config.vocab_size)
    model = TransformerLM(config.model_config()).to(device)
    if config.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = build_optimizer(model, config)
    amp_dtype = resolve_amp_dtype(device, config.amp_dtype)
    log_path = out_dir / "log.jsonl"
    best_valid_loss = math.inf
    running_tokens = 0
    start = time.time()

    for iteration in range(config.max_iters + 1):
        if iteration % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, valid_data, config, device, amp_dtype)
            is_best = losses["valid"] < best_valid_loss
            best_valid_loss = min(best_valid_loss, losses["valid"])
            record = {
                "iter": iteration,
                "train_loss": losses["train"],
                "valid_loss": losses["valid"],
                "best_valid_loss": best_valid_loss,
                "elapsed_sec": time.time() - start,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if is_best or iteration == config.max_iters:
                save_training_checkpoint(model, optimizer, config, iteration, best_valid_loss)

        if iteration == config.max_iters:
            break

        lr = cosine_lr(
            iteration,
            config.learning_rate,
            config.min_learning_rate,
            config.warmup_iters,
            config.max_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(config.grad_accum_steps):
            x, y = get_batch(train_data, config.batch_size, config.context_length, device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)
            (loss / config.grad_accum_steps).backward()
            total_loss += float(loss.item())
            running_tokens += config.batch_size * config.context_length
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if iteration % config.log_interval == 0:
            elapsed = max(time.time() - start, 1e-8)
            record = {
                "iter": iteration,
                "loss": total_loss / config.grad_accum_steps,
                "lr": lr,
                "tokens_per_sec": running_tokens / elapsed,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
