from __future__ import annotations

from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SERVER_ROOT.parent

A1_ROOT = WORKSPACE_ROOT / "assignment1-basics"
A2_ROOT = WORKSPACE_ROOT / "assignment2-systems"
A3_ROOT = WORKSPACE_ROOT / "assignment3-scaling"
A4_ROOT = WORKSPACE_ROOT / "assignment4-data"
A5_ROOT = WORKSPACE_ROOT / "assignment5-alignment"

STATE_DIR = SERVER_ROOT / "state"
EXPERIMENTS_DIR = SERVER_ROOT / "experiments"
EXPERIMENTS_JSON = STATE_DIR / "experiments.json"
AI_PROFILES_JSON = STATE_DIR / "ai_profiles.json"
GPU_TARGETS_JSON = STATE_DIR / "gpu_targets.json"
WEB_DIR = SERVER_ROOT / "web"
FRONTEND_DIST_DIR = SERVER_ROOT / "frontend" / "dist"


def python_bin(project_root: Path) -> Path:
    candidate = project_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path("python3")


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
