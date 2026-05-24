from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from cs336_systems.runtime import cuda_memory_snapshot


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    warmup: int
    repeats: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_sec: float | None
    cuda_memory: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_callable(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    repeats: int = 50,
    tokens_per_iter: int | None = None,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    synchronize_if_needed()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    durations_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        synchronize_if_needed()
        durations_ms.append((time.perf_counter() - start) * 1000)

    median_ms = statistics.median(durations_ms)
    tokens_per_sec = None
    if tokens_per_iter is not None:
        tokens_per_sec = tokens_per_iter / (median_ms / 1000)
    return BenchmarkResult(
        name=name,
        warmup=warmup,
        repeats=repeats,
        median_ms=median_ms,
        mean_ms=statistics.mean(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        tokens_per_sec=tokens_per_sec,
        cuda_memory=cuda_memory_snapshot(),
    )
