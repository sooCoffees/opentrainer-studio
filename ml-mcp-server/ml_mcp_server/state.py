from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import AI_PROFILES_JSON, EXPERIMENTS_JSON, GPU_TARGETS_JSON, ensure_dirs


@dataclass
class Experiment:
    id: str
    name: str
    created_at: float
    status: str = "created"
    run_dir: str | None = None
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GpuTarget:
    id: str
    name: str
    kind: str
    created_at: float
    status: str = "available"
    endpoint: str | None = None
    device: str | None = None
    ssh_host: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIProfile:
    id: str
    name: str
    purpose: str
    created_at: float
    status: str = "draft"
    gpu_target_id: str | None = None
    base_model: str | None = None
    dataset_path: str | None = None
    config_path: str | None = None
    experiment_id: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class JsonStore:
    item_type = dict

    def __init__(self, path: Path) -> None:
        ensure_dirs()
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {key: self.item_type(**value) for key, value in payload.items()}

    def save(self, items: dict[str, Any]) -> None:
        serializable = {key: asdict(value) for key, value in items.items()}
        self.path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")

    def upsert(self, item: Any) -> Any:
        items = self.load()
        items[item.id] = item
        self.save(items)
        return item

    def get(self, item_id: str) -> Any:
        items = self.load()
        if item_id not in items:
            raise KeyError(f"unknown item: {item_id}")
        return items[item_id]


class ExperimentStore:
    def __init__(self, path: Path = EXPERIMENTS_JSON) -> None:
        ensure_dirs()
        self.path = path

    def load(self) -> dict[str, Experiment]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {key: Experiment(**value) for key, value in payload.items()}

    def save(self, experiments: dict[str, Experiment]) -> None:
        serializable = {key: asdict(value) for key, value in experiments.items()}
        self.path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")

    def upsert(self, experiment: Experiment) -> Experiment:
        experiments = self.load()
        experiments[experiment.id] = experiment
        self.save(experiments)
        return experiment

    def create(
        self, experiment_id: str, name: str, metadata: dict[str, Any] | None = None
    ) -> Experiment:
        experiments = self.load()
        if experiment_id in experiments:
            raise ValueError(f"experiment already exists: {experiment_id}")
        experiment = Experiment(
            id=experiment_id,
            name=name,
            created_at=time.time(),
            metadata=metadata or {},
        )
        experiments[experiment_id] = experiment
        self.save(experiments)
        return experiment

    def get(self, experiment_id: str) -> Experiment:
        experiments = self.load()
        if experiment_id not in experiments:
            raise KeyError(f"unknown experiment: {experiment_id}")
        return experiments[experiment_id]


class GpuTargetStore(JsonStore):
    item_type = GpuTarget

    def __init__(self, path: Path = GPU_TARGETS_JSON) -> None:
        super().__init__(path)
        self._ensure_default_targets()

    def create(self, args: dict[str, Any]) -> GpuTarget:
        targets = self.load()
        target_id = args["id"]
        if target_id in targets:
            raise ValueError(f"gpu target already exists: {target_id}")
        target = GpuTarget(
            id=target_id,
            name=args["name"],
            kind=args.get("kind", "local"),
            endpoint=args.get("endpoint"),
            device=args.get("device"),
            ssh_host=args.get("ssh_host"),
            notes=args.get("notes", ""),
            metadata=args.get("metadata", {}),
            created_at=time.time(),
        )
        targets[target_id] = target
        self.save(targets)
        return target

    def _ensure_default_targets(self) -> None:
        targets = self.load()
        if targets:
            if "local-mps" in targets and targets["local-mps"].name == "Local Mac GPU":
                targets["local-mps"].name = "Local Machine"
                targets[
                    "local-mps"
                ].notes = "Current computer. Good for smoke tests and UI validation."
                self.save(targets)
            return
        now = time.time()
        targets["local-mps"] = GpuTarget(
            id="local-mps",
            name="Local Machine",
            kind="local",
            device="mps",
            created_at=now,
            notes="Current computer. Good for smoke tests and UI validation.",
        )
        targets["remote-template"] = GpuTarget(
            id="remote-template",
            name="Remote CUDA Server",
            kind="remote",
            endpoint="http://remote-host:8765",
            device="cuda",
            ssh_host="user@remote-host",
            created_at=now,
            status="template",
            notes="Fill in endpoint/SSH once you rent a GPU box.",
        )
        self.save(targets)


class AIProfileStore(JsonStore):
    item_type = AIProfile

    def __init__(self, path: Path = AI_PROFILES_JSON) -> None:
        super().__init__(path)

    def create(self, args: dict[str, Any]) -> AIProfile:
        profiles = self.load()
        profile_id = args["id"]
        if profile_id in profiles:
            raise ValueError(f"ai profile already exists: {profile_id}")
        profile = AIProfile(
            id=profile_id,
            name=args["name"],
            purpose=args.get("purpose", ""),
            gpu_target_id=args.get("gpu_target_id"),
            base_model=args.get("base_model"),
            dataset_path=args.get("dataset_path"),
            config_path=args.get("config_path"),
            notes=args.get("notes", ""),
            metadata=args.get("metadata", {}),
            created_at=time.time(),
        )
        profiles[profile_id] = profile
        self.save(profiles)
        return profile

    def update(self, profile_id: str, patch: dict[str, Any]) -> AIProfile:
        profiles = self.load()
        if profile_id not in profiles:
            raise KeyError(f"unknown ai profile: {profile_id}")
        profile = profiles[profile_id]
        for key, value in patch.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profiles[profile_id] = profile
        self.save(profiles)
        return profile

    def delete(self, profile_id: str) -> AIProfile:
        profiles = self.load()
        if profile_id not in profiles:
            raise KeyError(f"unknown ai profile: {profile_id}")
        profile = profiles.pop(profile_id)
        self.save(profiles)
        return profile
