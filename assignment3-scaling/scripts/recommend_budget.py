from __future__ import annotations

import argparse
import json

from cs336_scaling.offline.configs import make_model_spec, make_training_spec
from cs336_scaling.offline.fit import (
    best_points_by_compute,
    fit_compute_scaling_law,
    fit_parameter_frontier,
    load_isoflops_points,
)
from cs336_scaling.offline.flops import train_tokens_for_compute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoflops", default="data/isoflops_curves.json")
    parser.add_argument("--compute-budget", type=float, required=True)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    args = parser.parse_args()

    frontier = best_points_by_compute(load_isoflops_points(args.isoflops))
    loss_fit = fit_compute_scaling_law(frontier)
    param_coeff, param_exp = fit_parameter_frontier(frontier)
    predicted_params = int(param_coeff * (args.compute_budget**param_exp))
    model = make_model_spec(
        predicted_params, vocab_size=args.vocab_size, context_length=args.context_length
    )
    train_tokens = train_tokens_for_compute(args.compute_budget, predicted_params)
    spec = make_training_spec(model, train_tokens)
    print(
        json.dumps(
            {
                "compute_budget": args.compute_budget,
                "predicted_loss": loss_fit.predict(args.compute_budget),
                "predicted_params_from_frontier": predicted_params,
                "training_spec": spec.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
