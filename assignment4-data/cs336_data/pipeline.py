from __future__ import annotations

import os
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class TextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0
        self.block_tags = {"p", "div", "h1", "h2", "h3", "li", "ul", "br"}

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag == "li":
            self.parts.append("  * ")
        elif tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript"} and self.skip_depth > 0:
            self.skip_depth -= 1
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str):
        if self.skip_depth == 0:
            cleaned = re.sub(r"\s+", " ", data.strip())
            if cleaned:
                self.parts.append(cleaned)

    def text(self) -> str:
        text = " ".join(self.parts)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n\s+", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        lines = [line.strip() for line in re.sub(r"\n{2,}", "\n", text).splitlines() if line.strip()]
        normalized = []
        for line in lines:
            if line.startswith("*"):
                normalized.append("  • " + line[1:].strip())
            else:
                normalized.append(line)
        return "\n\n".join(normalized)


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    parser = TextHTMLParser()
    parser.feed(html_bytes.decode("utf-8", errors="ignore"))
    text = parser.text()
    return text if text.strip() else None


def identify_language(text: str) -> tuple[Any, float]:
    zh_chars = sum("\u4e00" <= ch <= "\u9fff" for ch in text)
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in text)
    total_letters = max(1, zh_chars + ascii_letters)
    if zh_chars / total_letters > 0.2:
        return "zh", zh_chars / total_letters
    return "en", ascii_letters / max(1, len(text))


def mask_emails(text: str) -> tuple[str, int]:
    return _mask(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text, "|||EMAIL_ADDRESS|||")


def mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r"(?<!\d)(?:\(\d{3}\)[ -]?|\d{3}[- ]?)\d{3}[- ]?\d{4}(?!\d)"
    return _mask(pattern, text, "|||PHONE_NUMBER|||")


def mask_ips(text: str) -> tuple[str, int]:
    pattern = r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
    return _mask(pattern, text, "|||IP_ADDRESS|||")


def _mask(pattern: str, text: str, replacement: str) -> tuple[str, int]:
    masked, count = re.subn(pattern, replacement, text)
    return masked, count


TOXIC_TERMS = {"fuck", "fucking", "fuckers", "twat", "moron", "idiot", "c*ck", "*ssh*le", "c*nts"}


def classify_nsfw(text: str) -> tuple[Any, float]:
    lowered = text.lower()
    score = sum(term in lowered for term in {"c*ck", "f*cking", "*ssh*le", "c*nts"})
    return ("nsfw" if score else "non-nsfw"), float(max(score, 1))


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    lowered = text.lower()
    score = sum(term in lowered for term in TOXIC_TERMS)
    return ("toxic" if score >= 3 else "non-toxic"), float(max(score, 1))


def classify_quality(text: str) -> tuple[Any, float]:
    lowered = text.lower()
    wiki_signals = ["reference", "history", "article", "retrieved", "isbn", "category", "encyclopedia"]
    cc_signals = [
        "cookie",
        "javascript",
        "click here",
        "privacy policy",
        "terms",
        "subscribe",
        "advertisement",
        "forum",
        "log in",
        "copyright",
        "powered by",
        "faq",
    ]
    wiki_score = sum(signal in lowered for signal in wiki_signals)
    cc_score = sum(signal in lowered for signal in cc_signals)
    if wiki_score >= cc_score:
        return "wiki", float(max(wiki_score, 1))
    return "cc", float(cc_score)


def gopher_quality_filter(text: str) -> bool:
    words = re.findall(r"\b\S+\b", text)
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not 50 <= len(alpha_words) <= 100_000:
        return False
    avg_len = sum(len(word) for word in alpha_words) / len(alpha_words)
    if avg_len < 3 or avg_len > 10:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and sum(line.endswith("...") or line.endswith("…") for line in lines) / len(lines) > 0.3:
        return False
    if sum(bool(re.search(r"[A-Za-z]", word)) for word in words) / max(1, len(words)) < 0.8:
        return False
    return True


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    line_counts: dict[str, int] = {}
    for input_file in input_files:
        seen_in_doc = set()
        for line in Path(input_file).read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if key and key not in seen_in_doc:
                line_counts[key] = line_counts.get(key, 0) + 1
                seen_in_doc.add(key)
    for input_file in input_files:
        lines = []
        for line in Path(input_file).read_text(encoding="utf-8").splitlines(keepends=True):
            key = line.strip()
            if key and line_counts.get(key, 0) > 1:
                continue
            lines.append(line)
        (output / Path(input_file).name).write_text("".join(lines), encoding="utf-8")


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    del num_hashes, num_bands
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[Path, str, set[str]]] = []
    for path_like in input_files:
        path = Path(path_like)
        text = path.read_text(encoding="utf-8")
        shingles = _word_ngrams(text, ngrams)
        if any(_jaccard(shingles, other) >= jaccard_threshold for _, _, other in kept):
            continue
        kept.append((path, text, shingles))
    for path, _text, _shingles in kept:
        shutil.copyfile(path, output / path.name)


def _word_ngrams(text: str, n: int) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))
