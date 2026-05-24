from __future__ import annotations

import argparse
import json

from cs336_scaling.offline.configs import make_isoflops_grid
from cs336_scaling.offline.flops import (
    estimate_training_flops,
    estimate_transformer_params,
)


def _ints(values: list[str]) -> list[int]:
    return [int(float(value)) for value in values]


def _floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", nargs="+", default=["1e7", "2.5e7", "5e7", "1e8"])
    parser.add_argument("--compute", nargs="+", default=["1e17", "3e17", "1e18"])
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    specs = make_isoflops_grid(
        parameter_targets=_ints(args.params),
        compute_budgets=_floats(args.compute),
        vocab_size=args.vocab_size,
        context_length=args.context_length,
    )
    payload = []
    for spec in specs:
        params = estimate_transformer_params(spec.model)
        payload.append(
            spec.to_dict()
            | {
                "estimated_params": params,
                "estimated_flops": estimate_training_flops(params, spec.train_tokens),
            }
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
