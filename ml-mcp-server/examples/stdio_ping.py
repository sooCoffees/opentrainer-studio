from __future__ import annotations

import json
import subprocess
import sys


proc = subprocess.Popen(
    [sys.executable, "-m", "ml_mcp_server.server", "--stdio"],
    text=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

assert proc.stdin is not None
assert proc.stdout is not None

proc.stdin.write(json.dumps({"id": 1, "method": "tools/list", "params": {}}) + "\n")
proc.stdin.flush()
print(proc.stdout.readline(), end="")

proc.stdin.write(
    json.dumps(
        {"id": 2, "method": "tools/call", "params": {"name": "system.report", "arguments": {}}}
    )
    + "\n"
)
proc.stdin.flush()
print(proc.stdout.readline(), end="")

proc.terminate()
