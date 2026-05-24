from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config: dict[str, object]
    status: str
    created_at: float
    metrics: dict[str, float]


class RunRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def create(
        self, run_id: str, config: dict[str, object], status: str = "planned"
    ) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            config=config,
            status=status,
            created_at=time.time(),
            metrics={},
        )
        self.append(record)
        return record

    def read_all(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(RunRecord(**json.loads(line)))
        return records
