from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .paths import FRONTEND_DIST_DIR, WEB_DIR
from .tools import ToolRegistry


def _ok(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def handle_rpc(registry: ToolRegistry, request: dict[str, Any]) -> dict[str, Any]:
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    try:
        if method == "tools/list":
            result = {"tools": registry.list_tools()}
        elif method == "tools/call":
            result = registry.call(params["name"], params.get("arguments") or {})
        elif method == "ping":
            result = {"ok": True}
        else:
            raise KeyError(f"unknown method: {method}")
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": str(exc), "type": type(exc).__name__},
        }


def run_stdio() -> None:
    registry = ToolRegistry()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = handle_rpc(registry, request)
        except Exception as exc:  # noqa: BLE001
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(exc), "type": type(exc).__name__},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


class Handler(BaseHTTPRequestHandler):
    registry = ToolRegistry()

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = _ok(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send(404, {"error": f"not found: {path.name}"})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_frontend_asset(self, request_path: str) -> None:
        root = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else WEB_DIR
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        candidate = (root / relative).resolve()
        root_resolved = root.resolve()
        if root_resolved not in candidate.parents and candidate != root_resolved:
            self._send(403, {"error": "forbidden"})
            return
        if candidate.exists() and candidate.is_file():
            self._send_file(candidate)
            return
        self._send_file(root / "index.html")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/health":
                self._send(200, {"ok": True, "service": "cs336-ml-mcp-server"})
            elif path == "/tools":
                self._send(200, {"tools": self.registry.list_tools()})
            elif path == "/experiments":
                self._send(200, self.registry.call("experiment.list", {}))
            elif path == "/ai":
                self._send(200, self.registry.call("ai.list", {}))
            elif path == "/gpu-targets":
                self._send(200, self.registry.call("gpu.list", {}))
            elif path == "/" or path.startswith("/assets/") or path in {"/app.js", "/style.css"}:
                self._send_frontend_asset(path)
            else:
                self._send_frontend_asset(path)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc), "type": type(exc).__name__})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if path == "/call":
                self._send(200, self.registry.call(payload["name"], payload.get("arguments") or {}))
            elif path == "/rpc":
                self._send(200, handle_rpc(self.registry, payload))
            else:
                self._send(404, {"error": f"unknown path: {path}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc), "type": type(exc).__name__})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[ml-mcp] " + fmt % args + "\n")


def run_http(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"ml-mcp-server listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stdio", action="store_true")
    mode.add_argument("--http", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.stdio:
        run_stdio()
    else:
        run_http(args.host, args.port)


if __name__ == "__main__":
    main()
