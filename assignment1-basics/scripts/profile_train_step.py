from __future__ import annotations

import argparse
import json

import torch

from cs336_basics.implementation import get_batch
from cs336_basics.io import read_token_file
from cs336_basics.model import TransformerLM
from cs336_basics.training import TrainConfig, build_optimizer, resolve_amp_dtype, resolve_device


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    device = resolve_device(config.device)
    amp_dtype = resolve_amp_dtype(device, config.amp_dtype)
    data = read_token_file(config.train_bin, config.vocab_size)
    model = TransformerLM(config.model_config()).to(device)
    if config.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = build_optimizer(model, config)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None and end is not None:
            start.record()
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(data, config.batch_size, config.context_length, device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if start is not None and end is not None:
            end.record()
            synchronize()
            timings.append(start.elapsed_time(end))
    payload = {
        "device": device,
        "steps": args.steps,
        "cuda_peak_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "cuda_step_ms": timings,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
