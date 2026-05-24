from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .http_client import http_json
from .paths import A1_ROOT, A2_ROOT, A3_ROOT, A4_ROOT, EXPERIMENTS_DIR, python_bin
from .processes import is_running, run_capture, spawn, terminate
from .state import AIProfileStore, ExperimentStore, GpuTargetStore

ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


def _json_output(result: dict[str, Any]) -> dict[str, Any]:
    stdout = result.get("stdout", "")
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = None
    return {"process": result, "json": parsed}


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": True,
        },
    }


class ToolRegistry:
    def __init__(self) -> None:
        self.store = ExperimentStore()
        self.ai_store = AIProfileStore()
        self.gpu_store = GpuTargetStore()
        self._tools: dict[str, tuple[dict[str, Any], ToolFn]] = {}
        self._register_all()

    def list_tools(self) -> list[dict[str, Any]]:
        return [schema for schema, _ in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name][1](arguments or {})

    def _register(self, schema: dict[str, Any], fn: ToolFn) -> None:
        self._tools[schema["name"]] = (schema, fn)

    def _register_all(self) -> None:
        self._register(
            _tool_schema(
                "experiment.create",
                "Create a tracked experiment record.",
                {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                ["id", "name"],
            ),
            self.experiment_create,
        )
        self._register(
            _tool_schema("experiment.list", "List tracked experiments."), self.experiment_list
        )
        self._register(
            _tool_schema(
                "experiment.status",
                "Return experiment state and process liveness.",
                {"id": {"type": "string"}},
                ["id"],
            ),
            self.experiment_status,
        )
        self._register(
            _tool_schema("system.report", "Run the A2 system/GPU report."), self.system_report
        )
        self._register(_tool_schema("ai.list", "List managed AI profiles."), self.ai_list)
        self._register(
            _tool_schema(
                "ai.create",
                "Create a managed AI profile with its own purpose, config, and GPU target.",
                {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "gpu_target_id": {"type": "string"},
                    "base_model": {"type": "string"},
                    "dataset_path": {"type": "string"},
                    "config_path": {"type": "string"},
                    "notes": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                ["id", "name"],
            ),
            self.ai_create,
        )
        self._register(
            _tool_schema(
                "ai.update",
                "Update a managed AI profile.",
                {
                    "id": {"type": "string"},
                    "patch": {"type": "object"},
                },
                ["id", "patch"],
            ),
            self.ai_update,
        )
        self._register(
            _tool_schema(
                "ai.delete",
                "Delete a managed AI profile. The linked experiment record is kept.",
                {
                    "id": {"type": "string"},
                },
                ["id"],
            ),
            self.ai_delete,
        )
        self._register(
            _tool_schema(
                "ai.assign_gpu",
                "Assign an AI profile to a local or remote GPU target.",
                {
                    "ai_id": {"type": "string"},
                    "gpu_target_id": {"type": "string"},
                },
                ["ai_id", "gpu_target_id"],
            ),
            self.ai_assign_gpu,
        )
        self._register(
            _tool_schema("gpu.list", "List local and remote GPU targets."), self.gpu_list
        )
        self._register(
            _tool_schema(
                "gpu.create",
                "Create a GPU target for local, rented, or remote training.",
                {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "endpoint": {"type": "string"},
                    "device": {"type": "string"},
                    "ssh_host": {"type": "string"},
                    "notes": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                ["id", "name"],
            ),
            self.gpu_create,
        )
        self._register(
            _tool_schema(
                "gpu.check",
                "Check whether a GPU target endpoint is reachable.",
                {
                    "gpu_target_id": {"type": "string"},
                    "endpoint": {"type": "string"},
                },
            ),
            self.gpu_check,
        )
        self._register(
            _tool_schema(
                "scaling.recommend",
                "Ask A3 to recommend model/training specs for a compute budget.",
                {
                    "compute_budget": {"type": "number"},
                    "vocab_size": {"type": "integer"},
                    "context_length": {"type": "integer"},
                },
                ["compute_budget"],
            ),
            self.scaling_recommend,
        )
        self._register(
            _tool_schema(
                "tokenizer.train",
                "Run A1 BPE tokenizer training.",
                {
                    "input": {"type": "string"},
                    "out_dir": {"type": "string"},
                    "vocab_size": {"type": "integer"},
                    "special_token": {"type": "array", "items": {"type": "string"}},
                },
                ["input", "out_dir"],
            ),
            self.tokenizer_train,
        )
        self._register(
            _tool_schema(
                "training.start",
                "Start an A1 training process and attach it to an experiment.",
                {
                    "experiment_id": {"type": "string"},
                    "config_path": {"type": "string"},
                },
                ["experiment_id", "config_path"],
            ),
            self.training_start,
        )
        self._register(
            _tool_schema(
                "training.status",
                "Return running status for an experiment training process.",
                {"experiment_id": {"type": "string"}},
                ["experiment_id"],
            ),
            self.training_status,
        )
        self._register(
            _tool_schema(
                "training.stop",
                "Terminate an experiment training process.",
                {"experiment_id": {"type": "string"}},
                ["experiment_id"],
            ),
            self.training_stop,
        )
        self._register(
            _tool_schema(
                "training.metrics",
                "Read the last rows from an experiment metrics.jsonl file.",
                {
                    "experiment_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                ["experiment_id"],
            ),
            self.training_metrics,
        )
        self._register(
            _tool_schema(
                "data.filter_text",
                "Run A4 language, PII, quality, and toxicity filters on text.",
                {"text": {"type": "string"}},
                ["text"],
            ),
            self.data_filter_text,
        )
        self._register(
            _tool_schema(
                "data.ingest_document",
                "Inspect or extract a local document for training data. Text files are extracted directly; PDF/OCR modes report required backends when unavailable.",
                {
                    "path": {"type": "string"},
                    "mode": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                ["path"],
            ),
            self.data_ingest_document,
        )
        self._register(
            _tool_schema(
                "cpp.health",
                "Call cpp-ai-service health endpoint.",
                {"base_url": {"type": "string"}},
            ),
            self.cpp_health,
        )
        self._register(
            _tool_schema(
                "cpp.generate",
                "Call cpp-ai-service generate endpoint.",
                {
                    "base_url": {"type": "string"},
                    "prompt": {"type": "string"},
                    "model": {"type": "string"},
                },
                ["prompt"],
            ),
            self.cpp_generate,
        )
        self._register(
            _tool_schema(
                "cpp.register_model",
                "Call cpp-ai-service model registration endpoint.",
                {
                    "base_url": {"type": "string"},
                    "model_id": {"type": "string"},
                    "path": {"type": "string"},
                    "backend": {"type": "string"},
                },
                ["model_id", "path"],
            ),
            self.cpp_register_model,
        )

    def experiment_create(self, args: dict[str, Any]) -> dict[str, Any]:
        experiment = self.store.create(args["id"], args["name"], args.get("metadata"))
        run_dir = EXPERIMENTS_DIR / experiment.id
        run_dir.mkdir(parents=True, exist_ok=True)
        experiment.run_dir = str(run_dir)
        self.store.upsert(experiment)
        return {"experiment": asdict(experiment)}

    def experiment_list(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"experiments": [asdict(exp) for exp in self.store.load().values()]}

    def experiment_status(self, args: dict[str, Any]) -> dict[str, Any]:
        experiment = self.store.get(args["id"])
        running = is_running(experiment.pid)
        if experiment.pid and not running and experiment.status == "running":
            experiment.status = "exited"
            self.store.upsert(experiment)
        return {"experiment": asdict(experiment), "running": running}

    def system_report(self, _args: dict[str, Any]) -> dict[str, Any]:
        py = python_bin(A2_ROOT)
        result = run_capture([str(py), "-m", "scripts.system_report"], cwd=A2_ROOT)
        return _json_output(result)

    def ai_list(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"ai_profiles": [asdict(profile) for profile in self.ai_store.load().values()]}

    def ai_create(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.ai_store.create(args)
        if not profile.experiment_id:
            experiment = self.store.create(
                f"ai-{profile.id}",
                f"{profile.name} training",
                {"ai_profile_id": profile.id, "gpu_target_id": profile.gpu_target_id},
            )
            run_dir = EXPERIMENTS_DIR / experiment.id
            run_dir.mkdir(parents=True, exist_ok=True)
            experiment.run_dir = str(run_dir)
            self.store.upsert(experiment)
            profile.experiment_id = experiment.id
            self.ai_store.upsert(profile)
        return {"ai_profile": asdict(profile)}

    def ai_update(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.ai_store.update(args["id"], args["patch"])
        return {"ai_profile": asdict(profile)}

    def ai_delete(self, args: dict[str, Any]) -> dict[str, Any]:
        profile = self.ai_store.delete(args["id"])
        return {
            "deleted": True,
            "ai_profile": asdict(profile),
            "kept_experiment_id": profile.experiment_id,
        }

    def ai_assign_gpu(self, args: dict[str, Any]) -> dict[str, Any]:
        self.gpu_store.get(args["gpu_target_id"])
        profile = self.ai_store.update(args["ai_id"], {"gpu_target_id": args["gpu_target_id"]})
        return {"ai_profile": asdict(profile)}

    def gpu_list(self, _args: dict[str, Any]) -> dict[str, Any]:
        return {"gpu_targets": [asdict(target) for target in self.gpu_store.load().values()]}

    def gpu_create(self, args: dict[str, Any]) -> dict[str, Any]:
        target = self.gpu_store.create(args)
        return {"gpu_target": asdict(target)}

    def gpu_check(self, args: dict[str, Any]) -> dict[str, Any]:
        target = None
        endpoint = args.get("endpoint")
        if args.get("gpu_target_id"):
            target = self.gpu_store.get(args["gpu_target_id"])
            endpoint = endpoint or target.endpoint
        if not endpoint:
            return {
                "reachable": False,
                "reason": "No endpoint configured. Add a URL like http://host:8765 for remote control.",
                "gpu_target": asdict(target) if target else None,
            }
        try:
            health = http_json("GET", f"{endpoint.rstrip('/')}/health", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            return {
                "reachable": False,
                "endpoint": endpoint,
                "reason": str(exc),
                "gpu_target": asdict(target) if target else None,
            }
        return {
            "reachable": True,
            "endpoint": endpoint,
            "health": health,
            "gpu_target": asdict(target) if target else None,
        }

    def scaling_recommend(self, args: dict[str, Any]) -> dict[str, Any]:
        py = python_bin(A3_ROOT)
        cmd = [
            str(py),
            "-m",
            "scripts.recommend_budget",
            "--compute-budget",
            str(args["compute_budget"]),
        ]
        if "vocab_size" in args:
            cmd.extend(["--vocab-size", str(args["vocab_size"])])
        if "context_length" in args:
            cmd.extend(["--context-length", str(args["context_length"])])
        return _json_output(run_capture(cmd, cwd=A3_ROOT))

    def tokenizer_train(self, args: dict[str, Any]) -> dict[str, Any]:
        py = python_bin(A1_ROOT)
        cmd = [
            str(py),
            "-m",
            "scripts.train_tokenizer",
            "--input",
            args["input"],
            "--out-dir",
            args["out_dir"],
            "--vocab-size",
            str(args.get("vocab_size", 1000)),
        ]
        for token in args.get("special_token", []):
            cmd.extend(["--special-token", token])
        return {"process": run_capture(cmd, cwd=A1_ROOT, timeout=int(args.get("timeout", 600)))}

    def training_start(self, args: dict[str, Any]) -> dict[str, Any]:
        experiment_id = args["experiment_id"]
        try:
            experiment = self.store.get(experiment_id)
        except KeyError:
            experiment = self.store.create(experiment_id, experiment_id)
            experiment.run_dir = str(EXPERIMENTS_DIR / experiment_id)

        if is_running(experiment.pid):
            return {"experiment": asdict(experiment), "running": True}

        run_dir = Path(experiment.run_dir or EXPERIMENTS_DIR / experiment_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        py = python_bin(A1_ROOT)
        proc = spawn(
            [str(py), "-m", "scripts.train_lm", "--config", args["config_path"]],
            cwd=A1_ROOT,
            log_path=run_dir / "train.log",
        )
        experiment.pid = proc.pid
        experiment.status = "running"
        experiment.run_dir = str(run_dir)
        experiment.metadata["config_path"] = args["config_path"]
        self.store.upsert(experiment)
        return {"experiment": asdict(experiment), "running": True}

    def training_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.experiment_status({"id": args["experiment_id"]})

    def training_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        experiment = self.store.get(args["experiment_id"])
        stopped = terminate(experiment.pid)
        if stopped:
            experiment.status = "stopping"
            self.store.upsert(experiment)
        return {"stopped": stopped, "experiment": asdict(experiment)}

    def training_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        experiment = self.store.get(args["experiment_id"])
        run_dir = Path(experiment.run_dir or "")
        limit = int(args.get("limit", 100))
        candidates = [
            run_dir / "metrics.jsonl",
            run_dir / "train_metrics.jsonl",
            Path(experiment.metadata.get("metrics_path", "")),
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return {"path": str(candidate), "metrics": _read_jsonl(candidate, limit)}
        return {"path": None, "metrics": []}

    def data_filter_text(self, args: dict[str, Any]) -> dict[str, Any]:
        sys.path.insert(0, str(A4_ROOT))
        from cs336_data.pipeline import (  # noqa: PLC0415
            classify_nsfw,
            classify_quality,
            classify_toxic_speech,
            gopher_quality_filter,
            identify_language,
            mask_emails,
            mask_ips,
            mask_phone_numbers,
        )

        text = args["text"]
        masked, email_count = mask_emails(text)
        masked, phone_count = mask_phone_numbers(masked)
        masked, ip_count = mask_ips(masked)
        return {
            "language": identify_language(masked),
            "quality": classify_quality(masked),
            "gopher_quality_pass": gopher_quality_filter(masked),
            "nsfw": classify_nsfw(masked),
            "toxic": classify_toxic_speech(masked),
            "masked_counts": {"emails": email_count, "phones": phone_count, "ips": ip_count},
            "masked_text": masked,
        }

    def data_ingest_document(self, args: dict[str, Any]) -> dict[str, Any]:
        path = Path(args["path"]).expanduser()
        mode = args.get("mode", "auto")
        if not path.exists():
            return {
                "ok": False,
                "stage": "not_found",
                "message": f"File does not exist: {path}",
                "next_step": "Put the file on this machine or mount the remote data folder, then retry.",
            }

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".jsonl", ".csv"}:
            text = path.read_text(encoding=args.get("encoding", "utf-8"), errors="replace")
            output_path = (
                Path(args["output_path"]).expanduser() if args.get("output_path") else None
            )
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
            return {
                "ok": True,
                "stage": "extracted",
                "source_type": suffix.lstrip("."),
                "characters": len(text),
                "preview": text[:1200],
                "output_path": str(output_path) if output_path else None,
                "next_step": "Use this text path as dataset_path on an AI profile, then choose a training config.",
            }

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore  # noqa: PLC0415
            except Exception:  # noqa: BLE001
                return {
                    "ok": False,
                    "stage": "backend_missing",
                    "source_type": "pdf",
                    "mode": mode,
                    "message": "PDF text extraction needs pypdf. Scanned PDFs need OCR backends such as pymupdf plus tesseract or a remote OCR service.",
                    "next_step": "Install a PDF/OCR backend on the training machine, or connect a remote GPU/data worker endpoint and run ingestion there.",
                }
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            output_path = (
                Path(args["output_path"]).expanduser() if args.get("output_path") else None
            )
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
            return {
                "ok": True,
                "stage": "extracted",
                "source_type": "pdf",
                "pages": len(pages),
                "characters": len(text),
                "preview": text[:1200],
                "output_path": str(output_path) if output_path else None,
                "next_step": "Review extracted text quality. If pages are blank, rerun with OCR backend.",
            }

        return {
            "ok": False,
            "stage": "unsupported_type",
            "source_type": suffix.lstrip(".") or "unknown",
            "message": "Supported now: txt, md, jsonl, csv, and PDF text extraction when pypdf is installed.",
            "next_step": "Convert the file to text/jsonl, or add a document parser plugin for this type.",
        }

    def cpp_health(self, args: dict[str, Any]) -> dict[str, Any]:
        base_url = args.get("base_url", "http://127.0.0.1:8080").rstrip("/")
        return http_json("GET", f"{base_url}/health")

    def cpp_generate(self, args: dict[str, Any]) -> dict[str, Any]:
        base_url = args.get("base_url", "http://127.0.0.1:8080").rstrip("/")
        payload = {"prompt": args["prompt"]}
        if "model" in args:
            payload["model"] = args["model"]
        return http_json("POST", f"{base_url}/generate", payload)

    def cpp_register_model(self, args: dict[str, Any]) -> dict[str, Any]:
        base_url = args.get("base_url", "http://127.0.0.1:8080").rstrip("/")
        payload = {
            "model_id": args["model_id"],
            "path": args["path"],
            "backend": args.get("backend", "python"),
            "registered_at": time.time(),
        }
        return http_json("POST", f"{base_url}/models/register", payload)
