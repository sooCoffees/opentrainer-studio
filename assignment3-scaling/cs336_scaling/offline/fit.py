from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IsoFLOPPoint:
    parameters: int
    compute_budget: float
    final_loss: float


@dataclass(frozen=True)
class IsoFLOPFrontier:
    compute_budget: float
    best_parameters: int
    best_loss: float


@dataclass(frozen=True)
class ScalingLawFit:
    irreducible_loss: float
    coefficient: float
    exponent: float
    rmse: float

    def predict(self, compute_budget: float) -> float:
        return self.irreducible_loss + self.coefficient * (
            compute_budget**self.exponent
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_isoflops_points(path: str | Path) -> list[IsoFLOPPoint]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [IsoFLOPPoint(**item) for item in payload]


def best_points_by_compute(points: list[IsoFLOPPoint]) -> list[IsoFLOPFrontier]:
    grouped: dict[float, list[IsoFLOPPoint]] = {}
    for point in points:
        grouped.setdefault(point.compute_budget, []).append(point)
    frontiers = []
    for compute_budget, group in sorted(grouped.items()):
        best = min(group, key=lambda item: item.final_loss)
        frontiers.append(
            IsoFLOPFrontier(compute_budget, best.parameters, best.final_loss)
        )
    return frontiers


def fit_compute_scaling_law(frontier: list[IsoFLOPFrontier]) -> ScalingLawFit:
    compute = np.asarray([point.compute_budget for point in frontier], dtype=np.float64)
    loss = np.asarray([point.best_loss for point in frontier], dtype=np.float64)
    best_fit: ScalingLawFit | None = None
    min_loss = float(loss.min())

    for irreducible in np.linspace(max(0.0, min_loss - 2.0), min_loss - 1e-4, 1000):
        residual = loss - irreducible
        if np.any(residual <= 0):
            continue
        x = np.log(compute)
        y = np.log(residual)
        exponent, log_coeff = np.polyfit(x, y, deg=1)
        coefficient = math.exp(float(log_coeff))
        pred = irreducible + coefficient * (compute**exponent)
        rmse = float(np.sqrt(np.mean((pred - loss) ** 2)))
        candidate = ScalingLawFit(
            float(irreducible), float(coefficient), float(exponent), rmse
        )
        if best_fit is None or candidate.rmse < best_fit.rmse:
            best_fit = candidate
    if best_fit is None:
        raise ValueError("could not fit scaling law")
    return best_fit


def fit_parameter_frontier(frontier: list[IsoFLOPFrontier]) -> tuple[float, float]:
    compute = np.asarray([point.compute_budget for point in frontier], dtype=np.float64)
    params = np.asarray([point.best_parameters for point in frontier], dtype=np.float64)
    exponent, log_coeff = np.polyfit(np.log(compute), np.log(params), deg=1)
    return float(math.exp(log_coeff)), float(exponent)
