from __future__ import annotations

import argparse
import json

from cs336_scaling.offline.fit import (
    best_points_by_compute,
    fit_compute_scaling_law,
    fit_parameter_frontier,
    load_isoflops_points,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/isoflops_curves.json")
    parser.add_argument("--predict-compute", type=float, action="append", default=[])
    args = parser.parse_args()

    points = load_isoflops_points(args.input)
    frontier = best_points_by_compute(points)
    loss_fit = fit_compute_scaling_law(frontier)
    param_coeff, param_exp = fit_parameter_frontier(frontier)
    predictions = [
        {
            "compute_budget": compute,
            "predicted_loss": loss_fit.predict(compute),
            "predicted_best_parameters": int(param_coeff * (compute**param_exp)),
        }
        for compute in args.predict_compute
    ]
    print(
        json.dumps(
            {
                "frontier": [point.__dict__ for point in frontier],
                "loss_fit": loss_fit.to_dict(),
                "parameter_frontier": {
                    "coefficient": param_coeff,
                    "exponent": param_exp,
                },
                "predictions": predictions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
