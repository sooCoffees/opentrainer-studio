from __future__ import annotations

import json
from urllib import request


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method=method,
        headers={"content-type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode("utf-8")
        if not text:
            return {"status": response.status}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"text": text}
        data.setdefault("status", response.status)
        return data
