import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class SummaryUnit:
    kind: str
    text: str


_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADER_RE = re.compile(r"^\s*(?:#+\s*)?[A-Z][A-Z /&()-]{1,40}:?\s*$")


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: PathLike, data: Any, *, indent: int = 2) -> Path:
    p = Path(path)
    if p.parent != Path("."):
        ensure_dir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    return p


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {p}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: PathLike, rows: Iterable[Any]) -> Path:
    p = Path(path)
    if p.parent != Path("."):
        ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def _normalize_summary_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.endswith(":") and len(stripped.split()) <= 6:
        return True
    return bool(_HEADER_RE.match(stripped))


def split_summary_sentences(text: str) -> List[str]:
    text = _normalize_summary_text(text)
    if not text:
        return []

    sentences: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if _is_header(block):
            continue
        parts = re.split(_SENT_BOUNDARY_RE, block)
        for part in parts:
            sent = part.strip()
            if sent:
                sentences.append(sent)
    return sentences


def split_summary_into_units(text: str) -> List[SummaryUnit]:
    text = _normalize_summary_text(text)
    if not text:
        return []

    units: List[SummaryUnit] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _is_header(line):
            units.append(SummaryUnit(kind="header", text=line))
            continue
        for sent in split_summary_sentences(line):
            units.append(SummaryUnit(kind="body", text=sent))
    return units


def join_summary_units(units: Sequence[SummaryUnit]) -> str:
    lines: List[str] = []
    current_body: List[str] = []

    def flush_body() -> None:
        nonlocal current_body
        if current_body:
            lines.append(" ".join(current_body).strip())
            current_body = []

    for unit in units:
        if unit.kind == "header":
            flush_body()
            lines.append(unit.text.strip())
        else:
            current_body.append(unit.text.strip())
    flush_body()
    return "\n".join(x for x in lines if x).strip()


__all__ = [
    "SummaryUnit",
    "ensure_dir",
    "join_summary_units",
    "read_json",
    "read_jsonl",
    "split_summary_into_units",
    "split_summary_sentences",
    "write_json",
    "write_jsonl",
]
