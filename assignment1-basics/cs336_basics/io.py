from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cs336_basics.implementation import BPETokenizer


def save_tokenizer(tokenizer: BPETokenizer, out_dir: str | os.PathLike) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    vocab_payload = {str(token_id): token_bytes.hex() for token_id, token_bytes in tokenizer.vocab.items()}
    merges_payload = [[left.hex(), right.hex()] for left, right in tokenizer.merges.keys()]
    payload = {
        "vocab": vocab_payload,
        "merges": merges_payload,
        "special_tokens": tokenizer.special_tokens,
    }
    (path / "tokenizer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_tokenizer(path: str | os.PathLike) -> BPETokenizer:
    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in payload["vocab"].items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
    return BPETokenizer(vocab, merges, payload.get("special_tokens"))


def save_json(payload: dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_token_file(token_ids: list[int], path: str | os.PathLike) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if max(token_ids, default=0) < 2**16 else np.uint32
    np.asarray(token_ids, dtype=dtype).tofile(output_path)


def read_token_file(path: str | os.PathLike, vocab_size: int) -> np.memmap:
    dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
    return np.memmap(path, dtype=dtype, mode="r")
