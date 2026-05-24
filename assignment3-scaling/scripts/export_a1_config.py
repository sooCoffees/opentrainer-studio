from __future__ import annotations

import argparse
import json
from pathlib import Path

from cs336_scaling.offline.configs import make_model_spec, make_training_spec
from cs336_scaling.offline.fit import (
    best_points_by_compute,
    fit_parameter_frontier,
    load_isoflops_points,
)
from cs336_scaling.offline.flops import train_tokens_for_compute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoflops", default="data/isoflops_curves.json")
    parser.add_argument("--compute-budget", type=float, required=True)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-tokens", type=int, default=131072)
    args = parser.parse_args()

    frontier = best_points_by_compute(load_isoflops_points(args.isoflops))
    param_coeff, param_exp = fit_parameter_frontier(frontier)
    predicted_params = int(param_coeff * (args.compute_budget**param_exp))
    model = make_model_spec(
        predicted_params, vocab_size=args.vocab_size, context_length=args.context_length
    )
    train_tokens = train_tokens_for_compute(args.compute_budget, predicted_params)
    spec = make_training_spec(model, train_tokens, batch_tokens=args.batch_tokens)
    batch_size = max(1, args.batch_tokens // model.context_length)
    max_iters = max(1, train_tokens // (batch_size * model.context_length))
    warmup_iters = max(1, spec.warmup_tokens // (batch_size * model.context_length))
    payload = {
        "train_bin": args.train_bin,
        "valid_bin": args.valid_bin,
        "out_dir": args.out_dir,
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "rope_theta": model.rope_theta,
        "attention_backend": spec.attention_backend,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "eval_interval": max(1, max_iters // 20),
        "eval_iters": 20,
        "log_interval": 10,
        "learning_rate": spec.learning_rate,
        "min_learning_rate": spec.min_learning_rate,
        "warmup_iters": warmup_iters,
        "weight_decay": spec.weight_decay,
        "grad_clip": spec.grad_clip,
        "grad_accum_steps": 1,
        "seed": 1337,
        "device": "auto",
        "compile": spec.compile,
        "amp_dtype": spec.amp_dtype,
        "optimizer": "fused_adamw",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "predicted_params": predicted_params,
                "config": payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
