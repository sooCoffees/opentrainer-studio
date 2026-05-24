from __future__ import annotations

import json
from urllib import error, request


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            if not text:
                return {"ok": True, "status": response.status}
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"text": text}
            data.setdefault("ok", 200 <= response.status < 300)
            data.setdefault("status", response.status)
            return data
    except error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"text": text}
        payload.update({"ok": False, "status": exc.code, "error": payload.get("error", exc.reason)})
        return payload
    except error.URLError as exc:
        return {
            "ok": False,
            "stage": "service_unreachable",
            "error": "C++ AI service is not reachable.",
            "detail": str(exc.reason),
            "next_step": "Start cpp-ai-service, confirm its port, then click Check Service again.",
        }
