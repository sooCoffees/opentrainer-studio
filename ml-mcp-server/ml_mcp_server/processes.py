from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


def is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def terminate(pid: int | None) -> bool:
    if pid is None or not is_running(pid):
        return False
    os.kill(pid, signal.SIGTERM)
    return True


def run_capture(args: list[str], cwd: Path, timeout: int = 120) -> dict:
    completed = subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def spawn(args: list[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    return subprocess.Popen(
        args,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
