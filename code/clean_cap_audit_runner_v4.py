from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from method_registry_v5_improved_v3 import METHODS

load_dotenv()

# ============================================================
# Basic text utilities
# ============================================================
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.",
    "pa.", "pa-c.", "np.", "rn.", "m.d.", "d.o.", "e.g.", "i.e."
}

KNOWN_HEADERS = {
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "HPI",
    "PAST HISTORY",
    "PAST MEDICAL HISTORY",
    "CURRENT MEDICATIONS",
    "SOCIAL HISTORY",
    "FAMILY HISTORY",
    "PHYSICAL EXAM",
    "PHYSICAL EXAMINATION",
    "RESULTS",
    "ASSESSMENT",
    "ASSESSMENT AND PLAN",
    "PLAN",
    "INSTRUCTIONS",
    "ROS",
    "REVIEW OF SYSTEMS",
    "DIAGNOSTIC TESTING",
}

CAP_CHUNK_METHODS = {"C", "D", "E"}
MAX_TRANSCRIPT_CHARS_PER_CHUNK = int(os.getenv("CAP_MAX_TRANSCRIPT_CHARS_PER_CHUNK", "3000"))
CHUNK_OVERLAP_TURNS = int(os.getenv("CAP_CHUNK_OVERLAP_TURNS", "2"))
MAX_CAP_PROPOSITIONS_PER_CHUNK = int(os.getenv("CAP_MAX_PROPOSITIONS_PER_CHUNK", "12"))
DETECTION_AXES = ("hallucination", "omission")
MEDSUM_PLANNING_CATEGORIES = (
    "Demographics and Social Determinants of Health",
    "Patient Intent",
    "Pertinent Positives",
    "Pertinent Negatives",
    "Pertinent Unknowns",
    "Medical History",
)


def _normalize_header_text(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().rstrip(":" )).upper()


def _is_header_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False

    norm = _normalize_header_text(t)
    if norm in KNOWN_HEADERS:
        return True

    letters = re.findall(r"[A-Za-z]", t)
    if not letters:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    if uppercase_ratio >= 0.9 and len(t) <= 60:
        return True
    if t.endswith(":") and len(t) <= 80 and uppercase_ratio >= 0.6:
        return True
    return False


def _protect_abbreviations(text: str):
    protected = text
    repls = {}
    for i, abbr in enumerate(sorted(ABBREVIATIONS, key=len, reverse=True)):
        token = f"__ABBR_{i}__"
        pattern = re.compile(re.escape(abbr), flags=re.IGNORECASE)
        if pattern.search(protected):
            repls[token] = abbr
            protected = pattern.sub(token, protected)
    return protected, repls


def _restore_abbreviations(text: str, repls: dict) -> str:
    out = text
    for token, abbr in repls.items():
        out = out.replace(token, abbr)
    return out


def _simple_sentence_split(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    protected, repls = _protect_abbreviations(t)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•\-])", protected)
    parts = [_restore_abbreviations(p.strip(), repls) for p in parts if p.strip()]
    return parts if parts else [t]


def _split_bullet_like_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"^\s*[*\-•]+\s*", "", raw).strip()
        if line:
            lines.append(line)
    if lines:
        return lines
    return [s.strip() for s in _simple_sentence_split(text) if s.strip()]


def _normalize_medsum_status(status_hint: str, fact_text: str) -> str:
    status = (status_hint or "").strip().lower()
    text = (fact_text or "").strip().lower()
    if status in {"present", "positive", "affirmed"}:
        return "Present"
    if status in {"absent", "negative", "negated"}:
        return "Absent"
    if status in {"unknown", "uncertain", "unsure"}:
        return "Unknown"
    if any(marker in text for marker in ["unknown", "unsure", "unclear", "not sure", "don't know", "doesn't know"]):
        return "Unknown"
    if text.startswith("no ") or " denies " in f" {text} " or " denied " in f" {text} ":
        return "Absent"
    return "Present"


def _infer_medsum_category(fact_text: str, status: str) -> str:
    text = (fact_text or "").strip().lower()
    if any(marker in text for marker in ["year-old", "years old", "male", "female", "works as", "lives with", "insurance", "smokes", "drinks alcohol", "housing", "job"]):
        return "Demographics and Social Determinants of Health"
    if any(marker in text for marker in ["here for", "seeking care", "presents for", "wants evaluation", "requests evaluation", "chief complaint", "comes in for"]):
        return "Patient Intent"
    if any(marker in text for marker in ["history of", "status post", "previous", "prior ", "family history", "allergic to", "allergy to", "appendectomy", "lumpectomy"]):
        return "Medical History"
    if status == "Absent":
        return "Pertinent Negatives"
    if status == "Unknown":
        return "Pertinent Unknowns"
    return "Pertinent Positives"


def _normalize_medsum_category(category_hint: str, fact_text: str, status: str) -> str:
    hint = re.sub(r"[\[\]]", "", (category_hint or "").strip()).lower()
    mapping = {
        "demographic": "Demographics and Social Determinants of Health",
        "demographics": "Demographics and Social Determinants of Health",
        "demographics and social determinants of health": "Demographics and Social Determinants of Health",
        "social determinants": "Demographics and Social Determinants of Health",
        "patient intent": "Patient Intent",
        "chief complaint": "Patient Intent",
        "pertinent positives": "Pertinent Positives",
        "positive": "Pertinent Positives",
        "pertinent negatives": "Pertinent Negatives",
        "negative": "Pertinent Negatives",
        "pertinent unknowns": "Pertinent Unknowns",
        "unknown": "Pertinent Unknowns",
        "medical history": "Medical History",
        "history": "Medical History",
        "pmh": "Medical History",
        "allergy": "Medical History",
    }
    if hint in mapping:
        return mapping[hint]
    return _infer_medsum_category(fact_text, status)


def _medsum_concept_key(fact_text: str) -> str:
    text = (fact_text or "").lower()
    text = re.sub(r"^(the patient|the clinician)\s+", "", text)
    text = re.sub(r"^(reports|states|notes|denies|has|have|had|is|are|was|were|plans|recommends|orders|prescribes)\s+", "", text)
    text = re.sub(r"\b(history of|possible|likely|currently|today|yesterday|tomorrow|for the past)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_medsum_fact_line(line: str) -> dict:
    text = line.strip()
    category_hint = ""
    status_hint = ""
    bracket_pattern = re.compile(r"^\[(Category:\s*)?([^\]]+)\]\s*")
    status_pattern = re.compile(r"^\[(Status:\s*)?([^\]]+)\]\s*")

    match = bracket_pattern.match(text)
    if match:
        category_hint = match.group(2).strip()
        text = text[match.end():].strip()
    match = status_pattern.match(text)
    if match:
        status_hint = match.group(2).strip()
        text = text[match.end():].strip()

    status = _normalize_medsum_status(status_hint, text)
    category = _normalize_medsum_category(category_hint, text, status)
    return {
        "category": category,
        "status": status,
        "text": text,
        "concept_key": _medsum_concept_key(text),
    }


def _resolve_medsum_unknowns(records: list[dict]) -> list[dict]:
    resolved_status_by_key: dict[str, str] = {}
    for rec in records:
        key = rec.get("concept_key") or ""
        status = rec.get("status") or ""
        if not key:
            continue
        if status in {"Present", "Absent"}:
            resolved_status_by_key[key] = status

    kept: list[dict] = []
    for rec in records:
        key = rec.get("concept_key") or ""
        status = rec.get("status") or ""
        if status == "Unknown" and key and resolved_status_by_key.get(key) in {"Present", "Absent"}:
            continue
        kept.append(rec)
    return kept


def postprocess_medsum_transcript_facts(raw_text: str) -> str:
    parsed_records = [_parse_medsum_fact_line(line) for line in _split_bullet_like_lines(raw_text)]
    parsed_records = [rec for rec in parsed_records if rec.get("text")]
    parsed_records = _resolve_medsum_unknowns(parsed_records)

    deduped: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for rec in parsed_records:
        key = (rec.get("category", ""), rec.get("status", ""), rec.get("concept_key", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    lines: list[str] = []
    for rec in deduped:
        lines.append(f"- [Category: {rec['category']}] [Status: {rec['status']}] {rec['text']}")
    return "\n".join(lines)


def split_summary_body_sentences(summary: str) -> list[str]:
    lines = [ln.rstrip() for ln in (summary or "").splitlines()]
    body_sentences: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if _is_header_line(line):
            continue
        body_sentences.extend(_simple_sentence_split(line))
    return [s.strip() for s in body_sentences if s.strip()]


# ============================================================
# Config / client helpers
# ============================================================

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class EndpointConfig:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str]
    disable_response_format: bool


GLOBAL_PROVIDER = os.getenv("GLOBAL_PROVIDER", os.getenv("LLM_PROVIDER", "runpod")).strip().lower()
GLOBAL_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
GLOBAL_LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "google/medgemma-27b-text-it")
GLOBAL_RUNPOD_MODEL = os.getenv("RUNPOD_MODEL", "ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g")

GLOBAL_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GLOBAL_LOCAL_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "dummy")
GLOBAL_RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "dummy")

GLOBAL_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GLOBAL_LOCAL_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")


def _resolve_endpoint(prefix: str) -> EndpointConfig:
    provider = os.getenv(f"{prefix}_PROVIDER", GLOBAL_PROVIDER).strip().lower()
    if provider not in {"openai", "local", "runpod"}:
        raise ValueError(f"Unsupported provider for {prefix}: {provider}")

    if provider == "openai":
        model = os.getenv(f"{prefix}_MODEL", GLOBAL_OPENAI_MODEL)
        api_key = os.getenv(f"{prefix}_API_KEY", GLOBAL_OPENAI_API_KEY)
        base_url = (os.getenv(f"{prefix}_BASE_URL", "") or "").strip() or GLOBAL_OPENAI_BASE_URL
    elif provider == "local":
        model = os.getenv(f"{prefix}_MODEL", GLOBAL_LOCAL_MODEL)
        api_key = os.getenv(f"{prefix}_API_KEY", GLOBAL_LOCAL_API_KEY)
        base_url = (os.getenv(f"{prefix}_BASE_URL", "") or "").strip() or GLOBAL_LOCAL_BASE_URL
    else:
        model = os.getenv(f"{prefix}_MODEL", GLOBAL_RUNPOD_MODEL)
        api_key = os.getenv(f"{prefix}_API_KEY", GLOBAL_RUNPOD_API_KEY)
        base_url = (os.getenv(f"{prefix}_BASE_URL", "") or "").strip()
        if not base_url:
            pod_id = os.getenv("RUNPOD_POD_ID", "")
            port = os.getenv("RUNPOD_PORT", "40080")
            if not pod_id:
                raise RuntimeError("RUNPOD_PROVIDER requires EXTRACTION_BASE_URL/DETECTION_BASE_URL or RUNPOD_POD_ID")
            base_url = f"https://{pod_id}-{port}.proxy.runpod.net/v1"

    disable_response_format = _env_flag(f"{prefix}_DISABLE_RESPONSE_FORMAT", False)
    return EndpointConfig(provider, model, api_key, base_url, disable_response_format)


EXTRACTION_CFG = _resolve_endpoint("EXTRACTION")
DETECTION_CFG = _resolve_endpoint("DETECTION")
DEFAULT_MODEL = EXTRACTION_CFG.model
DISABLE_RESPONSE_FORMAT = _env_flag("DISABLE_RESPONSE_FORMAT", False)


def make_client_from_config(cfg: EndpointConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key or "dummy", base_url=cfg.base_url, timeout=180.0)


# ============================================================
# JSON helpers
# ============================================================

def empty_cap_result() -> dict:
    return {
        "case_id": None,
        "atomic_propositions": [],
        "filtered_nonverifiable_units": [],
    }


def _compact_cap_record(item: Dict[str, Any], fallback_idx: int) -> Optional[Dict[str, Any]]:
    proposition_text = str(
        item.get("proposition_text")
        or item.get("fact_text")
        or item.get("text")
        or ""
    ).strip()
    if not proposition_text:
        return None

    cleaned = {
        "prop_id": str(item.get("prop_id") or f"P{fallback_idx}"),
        "category": str(item.get("category") or "").strip() or None,
        "speaker": str(item.get("speaker") or "").strip() or None,
        "predicate": str(item.get("predicate") or "").strip() or None,
        "status": str(item.get("status") or "").strip() or None,
        "temporality": str(item.get("temporality") or "").strip() or None,
        "claim_type_tags": item.get("claim_type_tags") if isinstance(item.get("claim_type_tags"), list) else [],
        "proposition_text": proposition_text,
    }
    return {k: v for k, v in cleaned.items() if v not in (None, [], "")}


def _normalize_cap_obj(obj: Any) -> dict:
    if not isinstance(obj, dict):
        return empty_cap_result()
    obj.setdefault("case_id", None)
    obj.setdefault("atomic_propositions", [])
    obj.setdefault("filtered_nonverifiable_units", [])
    props = obj["atomic_propositions"] if isinstance(obj["atomic_propositions"], list) else []
    normalized_props: List[Dict[str, Any]] = []
    for idx, item in enumerate(props, start=1):
        if not isinstance(item, dict):
            continue
        compact = _compact_cap_record(item, idx)
        if compact:
            normalized_props.append(compact)
    obj["atomic_propositions"] = normalized_props[:MAX_CAP_PROPOSITIONS_PER_CHUNK]

    filtered = obj["filtered_nonverifiable_units"] if isinstance(obj["filtered_nonverifiable_units"], list) else []
    normalized_filtered: List[str] = []
    for item in filtered:
        if isinstance(item, dict):
            unit_text = str(item.get("unit_text") or item.get("text") or "").strip()
            if unit_text:
                normalized_filtered.append(unit_text)
        elif isinstance(item, str) and item.strip():
            normalized_filtered.append(item.strip())
    obj["filtered_nonverifiable_units"] = normalized_filtered[:MAX_CAP_PROPOSITIONS_PER_CHUNK]
    return obj


def _cap_obj_is_suspicious(obj: dict) -> bool:
    props = obj.get("atomic_propositions") or []
    if not isinstance(props, list) or not props:
        return True
    for item in props:
        if not isinstance(item, dict):
            return True
        text = str(item.get("proposition_text") or "").strip()
        if not text:
            return True
        if len(text) > 400:
            return True
    return False


CAP_RETRY_SUFFIX = f"""
[Retry constraints]
- Return compact flat JSON only.
- Return at most {MAX_CAP_PROPOSITIONS_PER_CHUNK} atomic propositions.
- Include only these keys for each item:
  prop_id, category, speaker, predicate, status, temporality, claim_type_tags, proposition_text
- Do not include nested objects.
- `filtered_nonverifiable_units` must be a flat list of strings.
- If uncertain, return fewer propositions rather than malformed JSON.
""".strip()


def split_transcript_into_turn_chunks(
    transcript: str,
    *,
    max_chars: int = MAX_TRANSCRIPT_CHARS_PER_CHUNK,
    overlap_turns: int = CHUNK_OVERLAP_TURNS,
) -> List[str]:
    text = (transcript or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    turns = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not turns:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(turns):
        current: List[str] = []
        end = start
        while end < len(turns):
            candidate = "\n".join(current + [turns[end]])
            if current and len(candidate) > max_chars:
                break
            current.append(turns[end])
            end += 1
        if not current:
            current.append(turns[start])
            end = start + 1
        chunks.append("\n".join(current))
        if end >= len(turns):
            break
        start = max(end - max(0, overlap_turns), start + 1)
    return chunks


def merge_cap_results(cap_objects: List[dict]) -> dict:
    merged = empty_cap_result()
    seen = set()
    prop_idx = 1

    for obj in cap_objects:
        norm = _normalize_cap_obj(obj)
        for item in norm.get("atomic_propositions", []):
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("speaker") or "").strip().lower(),
                str(item.get("predicate") or "").strip().lower(),
                str(item.get("proposition_text") or item.get("surface_text") or "").strip().lower(),
                str(item.get("temporality") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            rec = dict(item)
            rec["prop_id"] = f"P{prop_idx}"
            prop_idx += 1
            merged["atomic_propositions"].append(rec)

        for item in norm.get("filtered_nonverifiable_units", []):
            if item not in merged["filtered_nonverifiable_units"]:
                merged["filtered_nonverifiable_units"].append(item)

    return merged


def safe_json_extract(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty response")

    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

    if s.startswith('"atomic_propositions"') or s.startswith('"filtered_nonverifiable_units"'):
        s = "{" + s + "}"

    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    if start != -1:
        end_positions = [m.start() for m in re.finditer(r"\}", s)]
        for end in reversed(end_positions):
            candidate = s[start:end + 1]
            if candidate.count("{") != candidate.count("}"):
                continue
            try:
                return json.loads(candidate)
            except Exception:
                continue

    raise ValueError(f"Could not extract valid JSON. Raw head:\n{s[:1200]}")


JSON_REPAIR_PROMPT = r"""
You will receive malformed JSON produced by a clinical NLP system.
Return one repaired valid JSON object only.
Do not add commentary, markdown, or code fences.
Preserve content whenever possible and remove unfinished trailing fragments if needed.
Return only this compact schema:
- atomic_propositions: list of up to {max_props} objects
- each object may contain only prop_id, category, speaker, predicate, status, temporality, claim_type_tags, proposition_text
- filtered_nonverifiable_units: flat list of strings

If the malformed content cannot be safely repaired into that schema, return:
{{"atomic_propositions": [], "filtered_nonverifiable_units": []}}

<Malformed JSON>
{bad_json}
</Malformed JSON>
""".strip()


CAP_RESULT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "atomic_propositions": {
            "type": "array",
            "maxItems": MAX_CAP_PROPOSITIONS_PER_CHUNK,
            "items": {
                "type": "object",
                "properties": {
                    "prop_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "category": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "predicate": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "status": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "temporality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "claim_type_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "proposition_text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["proposition_text"],
                "additionalProperties": False,
            },
        },
        "filtered_nonverifiable_units": {
            "type": "array",
            "maxItems": MAX_CAP_PROPOSITIONS_PER_CHUNK,
            "items": {"type": "string"},
        },
    },
    "required": ["atomic_propositions", "filtered_nonverifiable_units"],
    "additionalProperties": False,
}


def _repair_json_via_llm_if_needed(text: str) -> dict:
    prompt = JSON_REPAIR_PROMPT.format(
        bad_json=(text or "")[:12000],
        max_props=MAX_CAP_PROPOSITIONS_PER_CHUNK,
    )
    repaired_text = llm_generate(
        "repair",
        prompt,
        max_tokens=2048,
        force_json=True,
        json_schema=CAP_RESULT_JSON_SCHEMA,
    )
    repaired_obj = safe_json_extract(repaired_text)
    return repaired_obj if isinstance(repaired_obj, dict) else empty_cap_result()


def _extract_cap_once(prompt: str, *, max_tokens: int) -> tuple[str, dict]:
    text = llm_generate(
        "extraction",
        prompt,
        max_tokens=max_tokens,
        force_json=True,
        json_schema=CAP_RESULT_JSON_SCHEMA,
    )
    obj = _normalize_cap_obj(safe_json_extract(text))
    return text, obj


def _try_parse_json_string(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return text
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return text
    return text


# ============================================================
# LLM caller
# ============================================================

def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _chat(
    system_prompt: str,
    user_prompt: str,
    *,
    client: OpenAI,
    model: str,
    max_tokens: int,
    cfg: EndpointConfig,
    force_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    base_kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    structured_enabled = force_json and not (DISABLE_RESPONSE_FORMAT or cfg.disable_response_format)
    if not structured_enabled:
        resp = client.chat.completions.create(**base_kwargs)
        return _normalize_message_content(resp.choices[0].message.content)

    attempts: List[Dict[str, Any]] = []
    if json_schema:
        attempts.append(
            {
                **base_kwargs,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "cap_result",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            }
        )
        if cfg.provider in {"local", "runpod"}:
            attempts.append(
                {
                    **base_kwargs,
                    "response_format": {"type": "json_object"},
                    "extra_body": {
                        "structured_outputs": {
                            "json": json_schema,
                        }
                    },
                }
            )
    else:
        attempts.append({**base_kwargs, "response_format": {"type": "json_object"}})

    attempts.append(base_kwargs)

    last_exc: Optional[Exception] = None
    resp = None
    for idx, kwargs in enumerate(attempts, start=1):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_exc = exc
            if idx < len(attempts):
                print(f"[WARN] Structured output request attempt {idx} failed; trying fallback: {exc}")
            else:
                raise

    return _normalize_message_content(resp.choices[0].message.content)


def llm_generate(
    client_kind: str,
    user_prompt: str,
    *,
    max_tokens: Optional[int] = None,
    force_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> str:
    if client_kind == "detection":
        cfg = DETECTION_CFG
        max_tokens = max_tokens or 8192
    elif client_kind == "repair":
        cfg = EXTRACTION_CFG
        max_tokens = max_tokens or 8192
    else:
        cfg = EXTRACTION_CFG
        max_tokens = max_tokens or 8192

    client = make_client_from_config(cfg)
    system_prompt = "You are a precise clinical NLP system that follows the provided extraction and auditing instructions exactly."
    return _chat(
        system_prompt,
        user_prompt,
        client=client,
        model=cfg.model,
        max_tokens=max_tokens,
        cfg=cfg,
        force_json=force_json,
        json_schema=json_schema,
    )


# ============================================================
# Prompt helpers
# ============================================================

def format_kiwi_entities(entities: Any, max_items: int = 20) -> str:
    if entities is None:
        return "[]"
    if isinstance(entities, str):
        return entities
    compact = []
    for ent in (entities or [])[:max_items]:
        if not isinstance(ent, dict):
            continue
        compact_item = {
            "entity": ent.get("entity") or ent.get("text"),
            "type": ent.get("type"),
            "line": ent.get("line"),
            "sentence": ent.get("sentence"),
        }
        mappings = ent.get("mapping") or []
        if mappings:
            top = mappings[0]
            compact_item["top_mapping"] = {
                "term": top.get("term"),
                "concept_id": top.get("concept_id"),
                "semantic_type": top.get("semantic_type"),
            }
        compact.append(compact_item)
    return json.dumps(compact, ensure_ascii=False, indent=2)


def format_kiwi_relations(relations: Any, max_items: int = 15) -> str:
    if relations is None:
        return "[]"
    if isinstance(relations, str):
        return relations
    compact = []
    for rel in (relations or [])[:max_items]:
        if not isinstance(rel, dict):
            continue
        entity1 = rel.get("entity1") or {}
        entity2 = rel.get("entity2") or {}
        compact.append(
            {
                "entity1_text": entity1.get("text"),
                "entity1_type": entity1.get("type"),
                "entity2_text": entity2.get("text"),
                "entity2_type": entity2.get("type"),
                "relation": rel.get("relation", rel.get("type", "related_to")),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def build_prompt(
    method_key: str,
    prompt_key: str,
    *,
    transcript: str = "",
    summary_text: str = "",
    indexed_summary_sentences: str = "",
    indexed_facts: str = "",
    indexed_summary_facts: str = "",
    entities: Any = None,
    relations: Any = None,
) -> str:
    template = METHODS[method_key][prompt_key]
    return template.format(
        transcript=transcript,
        summary_text=summary_text,
        indexed_summary_sentences=indexed_summary_sentences,
        indexed_facts=indexed_facts,
        indexed_summary_facts=indexed_summary_facts,
        entities=format_kiwi_entities(entities),
        relations=format_kiwi_relations(relations),
    )


def build_extract_prompt(method_key: str, transcript: str, *, entities: Any = None, relations: Any = None) -> str:
    return build_prompt(method_key, "extract_prompt", transcript=transcript, entities=entities, relations=relations)


def build_summary_extract_prompt(method_key: str, full_summary: str) -> tuple[str, list[str]]:
    summary_sentences = split_summary_body_sentences(full_summary)
    indexed_summary_sentences = format_indexed_sentences(summary_sentences, prefix="S")
    prompt = build_prompt(
        method_key,
        "summary_extract_prompt",
        summary_text=full_summary,
        indexed_summary_sentences=indexed_summary_sentences,
    )
    return prompt, summary_sentences


# ============================================================
# Kiwi helpers
# ============================================================
HIGH_SEVERITY_KEYWORDS = [
    "diagnosis", "diagnosed", "impression", "radiculopathy", "neuropathy",
    "eliquis", "apixaban", "blood thinner", "anticoagulant", "allergy", "prednisone",
    "gabapentin", "mri", "x-ray", "ct", "hemoglobin", "weakness", "numbness",
    "tingling", "referral", "follow up", "follow-up", "return precaution", "epidural",
    "injection", "surgery", "start", "continue", "stop",
]


def _iter_kiwi_entities(entities: Any) -> list[dict]:
    if entities is None:
        return []
    if isinstance(entities, str):
        parsed = _try_parse_json_string(entities)
        if parsed is not entities:
            return _iter_kiwi_entities(parsed)
        return []
    return [x for x in entities if isinstance(x, dict)] if isinstance(entities, list) else []


def _iter_kiwi_relations(relations: Any) -> list[dict]:
    if relations is None:
        return []
    if isinstance(relations, str):
        parsed = _try_parse_json_string(relations)
        if parsed is not relations:
            return _iter_kiwi_relations(parsed)
        return []
    return [x for x in relations if isinstance(x, dict)] if isinstance(relations, list) else []


def _text_has_any(text: str, needles: list[str]) -> bool:
    t = (text or "").lower()
    return any(n in t for n in needles)


def _is_high_salience_kiwi_entity(ent: dict) -> bool:
    etype = str(ent.get("type") or "").lower()
    entity = str(ent.get("entity") or ent.get("text") or "")
    mapping = (ent.get("mapping") or [])
    top = mapping[0] if isinstance(mapping, list) and mapping else {}
    sem = str(top.get("semantic_type") or "").lower()
    text = f"{etype} {entity} {top.get('term','')} {sem}".lower()
    return any(k in text for k in HIGH_SEVERITY_KEYWORDS) or any(k in etype for k in ["drug", "problem", "test", "treatment"])


def _is_high_salience_kiwi_relation(rel: dict) -> bool:
    relation = str(rel.get("relation") or rel.get("type") or "").lower()
    e1 = rel.get("entity1") or {}
    e2 = rel.get("entity2") or {}
    text = f"{relation} {e1.get('text','')} {e2.get('text','')}".lower()
    return any(k in text for k in HIGH_SEVERITY_KEYWORDS)


def _extract_high_salience_kiwi_signals(entities: Any, relations: Any, max_items: int = 12) -> list[dict]:
    signals = []
    for ent in _iter_kiwi_entities(entities):
        if _is_high_salience_kiwi_entity(ent):
            mappings = ent.get("mapping") or []
            top = mappings[0] if mappings else {}
            signals.append(
                {
                    "kind": "entity",
                    "text": str(ent.get("entity") or ent.get("text") or ""),
                    "type": str(ent.get("type") or ""),
                    "mapping_term": str(top.get("term") or ""),
                    "semantic_type": str(top.get("semantic_type") or ""),
                }
            )
    for rel in _iter_kiwi_relations(relations):
        if _is_high_salience_kiwi_relation(rel):
            e1 = rel.get("entity1") or {}
            e2 = rel.get("entity2") or {}
            signals.append(
                {
                    "kind": "relation",
                    "text": f"{e1.get('text','')} --{rel.get('relation', rel.get('type','related_to'))}--> {e2.get('text','')}",
                    "type": str(rel.get("relation") or rel.get("type") or ""),
                    "mapping_term": "",
                    "semantic_type": "",
                }
            )
    dedup = []
    seen = set()
    for sig in signals:
        key = (sig["kind"], sig["text"], sig["type"], sig["mapping_term"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(sig)
    return dedup[:max_items]


def _coverage_text_from_cap_obj(cap_obj: Any) -> str:
    records = normalize_detection_fact_records(cap_obj)
    pieces = []
    for rec in records:
        for key in ["fact_text", "proposition_text"]:
            val = rec.get(key)
            if val:
                pieces.append(str(val).lower())
    return "\n".join(pieces)


def _signal_covered_by_cap(signal: dict, cap_obj: Any) -> bool:
    cap_text = _coverage_text_from_cap_obj(cap_obj)
    text = str(signal.get("text") or "").lower().strip()
    mapping_term = str(signal.get("mapping_term") or "").lower().strip()
    for cand in [text, mapping_term]:
        if cand and len(cand) >= 4 and cand in cap_text:
            return True
    toks = [t for t in re.findall(r"[a-z0-9]+", mapping_term or text) if len(t) > 3]
    return bool(toks and sum(t in cap_text for t in toks) >= min(2, len(toks)))


def _signal_relevant_to_transcript_chunk(signal: dict, transcript: str) -> bool:
    t = (transcript or "").lower()
    text = str(signal.get("text") or "").lower().strip()
    mapping_term = str(signal.get("mapping_term") or "").lower().strip()
    candidates = [c for c in [text, mapping_term] if c]
    for cand in candidates:
        if len(cand) >= 5 and cand in t:
            return True
        toks = [tok for tok in re.findall(r"[a-z0-9]+", cand) if len(tok) > 3]
        if toks and sum(tok in t for tok in toks) >= min(2, len(toks)):
            return True
    return False


def load_kiwi_mapping_for_case(kiwi_mapping_dir: str | Path, case_id: str) -> Optional[dict]:
    kiwi_dir = Path(kiwi_mapping_dir)
    if not kiwi_dir.exists() or not kiwi_dir.is_dir():
        return None
    candidates = list(kiwi_dir.glob(f"{case_id}*_result.json"))
    if not candidates:
        return None
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f)


def get_case_payload_from_kiwi_mapping(kiwi_json: dict, case_id: str) -> Optional[dict]:
    if not kiwi_json:
        return None
    for key, value in kiwi_json.items():
        if case_id in str(key):
            return value
    return None


KIWI_COVERAGE_AUDIT_REPAIR_PROMPT_V3 = r"""
You are revising a CAP extraction for coverage completeness.

Goal:
Use Kiwi IE hints as a recall-oriented auxiliary channel for high-severity coverage audit.
The transcript remains the only source of truth.
The current CAP extraction may have missed some high-salience signals.

Instructions:
1. Read the transcript and the current CAP JSON.
2. Review the candidate high-salience Kiwi signals that are not yet well reflected in the current CAP extraction.
3. Add only those missing propositions that are clearly grounded in the transcript.
4. Do NOT add propositions supported only by Kiwi and not by the transcript.
5. Preserve the existing CAP schema and valid JSON structure.
6. Prefer missing propositions involving:
   - diagnosis / impression
   - medication exposure or final medication state
   - anticoagulation / allergy / safety-critical exposure
   - abnormal result / numeric value
   - procedure / referral / test order / treatment plan
   - follow-up timing / return precaution / patient-facing instruction
   - active symptom burden / persistence / progression
7. Keep one CAP = one clinical assertion.
8. Do not over-specify clinician identity or generate unsupported normalization.
9. If no transcript-grounded repair is needed, return the original CAP JSON unchanged.
10. {mapping_instruction}
11. Keep the output compact and flat.
12. Return at most {max_props} atomic propositions.
13. Each atomic proposition may contain only:
    prop_id, category, speaker, predicate, status, temporality, claim_type_tags, proposition_text
14. `filtered_nonverifiable_units` must be a flat list of strings.

Return exactly one JSON object and nothing else.
Do not include markdown, code fences, commentary, or trailing explanation.
Keep the JSON compact.

<Input Transcript>
{transcript}
</Input Transcript>

<Current CAP JSON>
{current_cap_json}
</Current CAP JSON>

<Missing High-Salience Kiwi Signals>
{missing_signals_json}
</Missing High-Salience Kiwi Signals>
""".strip()


def _repair_cap_with_kiwi_coverage_audit(
    transcript: str,
    cap_obj: dict,
    entities: Any,
    relations: Any,
    *,
    allow_mapping: bool,
) -> dict:
    signals = _extract_high_salience_kiwi_signals(entities, relations)
    signals = [sig for sig in signals if _signal_relevant_to_transcript_chunk(sig, transcript)]
    missing = [sig for sig in signals if not _signal_covered_by_cap(sig, cap_obj)]
    if not missing:
        return cap_obj

    user_prompt = KIWI_COVERAGE_AUDIT_REPAIR_PROMPT_V3.format(
        transcript=transcript,
        current_cap_json=json.dumps(cap_obj, ensure_ascii=False),
        missing_signals_json=json.dumps(missing[:8], ensure_ascii=False),
        max_props=MAX_CAP_PROPOSITIONS_PER_CHUNK,
        mapping_instruction=(
            "You may use ontology-aware normalization only when clinically plausible and clearly transcript-grounded."
            if allow_mapping
            else "Do NOT use ontology-aware mapping. Normalize only to transcript-grounded plain clinical English."
        ),
    )
    try:
        attempts = [
            user_prompt,
            user_prompt + "\n\n" + CAP_RETRY_SUFFIX,
        ]
        last_text = ""
        for prompt in attempts:
            last_text = llm_generate(
                "repair",
                prompt,
                max_tokens=2048,
                force_json=True,
                json_schema=CAP_RESULT_JSON_SCHEMA,
            )
            repaired = _normalize_cap_obj(safe_json_extract(last_text))
            if not _cap_obj_is_suspicious(repaired):
                return repaired
        if last_text:
            repaired = _repair_json_via_llm_if_needed(last_text)
            repaired = _normalize_cap_obj(repaired)
            if not _cap_obj_is_suspicious(repaired):
                return repaired
        return cap_obj
    except Exception as exc:
        print(f"[WARN] Kiwi coverage audit repair failed: {exc}")
        return cap_obj


# ============================================================
# Extraction
# ============================================================

def extract_method_a_bn_atomic_facts(transcript: str, *, model: str = DEFAULT_MODEL) -> str:
    user_prompt = build_extract_prompt("A", transcript)
    raw = llm_generate("extraction", user_prompt, max_tokens=8192)
    return postprocess_medsum_transcript_facts(raw)


def extract_method_b_verifact_atomic_claims(transcript: str, *, model: str = DEFAULT_MODEL) -> str:
    user_prompt = build_extract_prompt("B", transcript)
    return llm_generate("extraction", user_prompt, max_tokens=8192)


def extract_method_c_cap_only(transcript: str, *, model: str = DEFAULT_MODEL) -> dict:
    user_prompt = build_extract_prompt("C", transcript)
    attempts = [
        user_prompt,
        user_prompt + "\n\n" + CAP_RETRY_SUFFIX,
        user_prompt + "\n\n" + CAP_RETRY_SUFFIX + "\nReturn fewer propositions if needed.",
    ]
    last_text = ""
    last_exc: Optional[Exception] = None
    for prompt in attempts:
        try:
            last_text, obj = _extract_cap_once(prompt, max_tokens=2048)
            if not _cap_obj_is_suspicious(obj):
                return obj
            last_exc = ValueError("Suspicious CAP output")
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] C extraction parse failed: {exc}")
    if last_text:
        try:
            repaired = _repair_json_via_llm_if_needed(last_text)
            repaired = _normalize_cap_obj(repaired)
            if not _cap_obj_is_suspicious(repaired):
                return repaired
        except Exception:
            pass
    return empty_cap_result()


def extract_method_d_cap_kiwi_no_umls(transcript: str, entities: Any, relations: Any, *, model: str = DEFAULT_MODEL) -> dict:
    base_obj = extract_method_c_cap_only(transcript, model=model)
    return _repair_cap_with_kiwi_coverage_audit(transcript, base_obj, entities, relations, allow_mapping=False)


def extract_method_e_cap_kiwi_umls(transcript: str, entities: Any, relations: Any, *, model: str = DEFAULT_MODEL) -> dict:
    base_obj = extract_method_c_cap_only(transcript, model=model)
    return _repair_cap_with_kiwi_coverage_audit(transcript, base_obj, entities, relations, allow_mapping=True)


extract_method_b_kiwi_statement = extract_method_b_verifact_atomic_claims
extract_method_c_ours_cap = extract_method_c_cap_only
extract_method_d_ours_cap_kiwi = extract_method_d_cap_kiwi_no_umls
extract_method_e_kiwi_only = extract_method_e_cap_kiwi_umls


def extract_cap_with_chunking(
    method_key: str,
    transcript: str,
    *,
    model: str = DEFAULT_MODEL,
    entities: Any = None,
    relations: Any = None,
) -> tuple[dict, List[float], int]:
    chunks = split_transcript_into_turn_chunks(transcript)
    if not chunks:
        return empty_cap_result(), [], 0

    part_times: List[float] = []

    if method_key not in CAP_CHUNK_METHODS or len(chunks) == 1:
        t0 = time.perf_counter()
        if method_key == "C":
            obj = extract_method_c_cap_only(transcript, model=model)
        elif method_key == "D":
            obj = extract_method_d_cap_kiwi_no_umls(transcript, entities, relations, model=model)
        elif method_key == "E":
            obj = extract_method_e_cap_kiwi_umls(transcript, entities, relations, model=model)
        else:
            raise ValueError(f"Unsupported chunked CAP method: {method_key}")
        part_times.append(round(time.perf_counter() - t0, 3))
        return obj, part_times, 1

    chunk_results: List[dict] = []
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[INFO] {method_key} CAP extraction chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
        t0 = time.perf_counter()
        chunk_obj = extract_method_c_cap_only(chunk, model=model)
        if method_key == "D":
            chunk_obj = _repair_cap_with_kiwi_coverage_audit(
                chunk,
                chunk_obj,
                entities,
                relations,
                allow_mapping=False,
            )
        elif method_key == "E":
            chunk_obj = _repair_cap_with_kiwi_coverage_audit(
                chunk,
                chunk_obj,
                entities,
                relations,
                allow_mapping=True,
            )
        part_times.append(round(time.perf_counter() - t0, 3))
        chunk_results.append(chunk_obj)

    merged = merge_cap_results(chunk_results)

    return merged, part_times, len(chunks)


# ============================================================
# Detection helpers
# ============================================================
LOW_VALUE_HALLUCINATION_PATTERNS = [
    "pa-c", "m.d.", "md", "np", "provider", "later that evening", "went back to sleep",
    "2-3 hours", "2 to 3 hours", "tablet", "capsule", "review of systems", "hoffman",
    "sensation intact", "strength is", "reflexes", "throughout", "date of service",
]

EVENT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "without", "in", "on", "at", "by",
    "from", "that", "this", "these", "those", "patient", "clinician", "doctor", "provider",
    "reports", "report", "reported", "states", "state", "stated", "has", "have", "had", "is",
    "are", "was", "were", "be", "being", "been", "shows", "show", "reveals", "revealed",
    "demonstrates", "demonstrated", "notes", "note", "noted", "advised", "advises", "recommend",
    "recommended", "recommends", "plans", "planned", "plan", "orders", "ordered", "order",
    "current", "past", "history", "today", "yesterday", "tomorrow",
}


def _looks_numeric_or_result_fact(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "mri", "x-ray", "ct", "scan", "ultrasound", "imaging", "result", "reveals",
        "shows", "demonstrates", "level", "dose", "mg", "cm", "mm", "g/dl", "%",
        "disc degeneration", "arthropathy", "nerve root", "compression", "eliquis",
        "gabapentin", "prednisone", "radiculopathy", "neuropathy"
    ])


def _event_type_from_record(rec: dict) -> str:
    category = str(rec.get("category") or "").strip().lower()
    predicate = str(rec.get("predicate") or "").strip().lower()
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "").lower()

    if category in {"diagnosis", "assessment", "impression"}:
        return "DiagnosisImpression"
    if category in {"medicationplan", "medicationexposure", "medication"}:
        if predicate in {"prescribes", "orders", "recommends", "plans", "instructs"}:
            return "MedicationPlan"
        return "MedicationState"
    if category in {"followupplan"}:
        return "FollowUpSafety"
    if category in {"testplan", "procedure"}:
        return "TestProcedurePlan"
    if category in {"finding", "result"}:
        return "FindingResult"
    if category in {"allergy"}:
        return "MedicationState"
    if category in {"chiefcomplaint", "symptom"}:
        return "SymptomState"
    if category in {"demographic"}:
        return "PMFSContext"
    if "follow-up" in text or "follow up" in text or "return precaution" in text:
        return "FollowUpSafety"
    if any(k in text for k in ["allergy", "allergic", "anticoag", "eliquis", "warfarin"]):
        return "MedicationState"
    if _looks_numeric_or_result_fact(text):
        return "FindingResult"
    if any(k in text for k in ["diagnosis", "impression", "likely", "consistent with"]):
        return "DiagnosisImpression"
    if any(k in text for k in ["prescribe", "start ", "stop ", "continue ", "hold ", "take "]):
        return "MedicationState"
    if any(k in text for k in ["refer", "follow-up", "follow up", "return", "come back", "schedule", "order"]):
        return "FollowUpSafety"
    return "PMFSContext"


def _event_priority_label(rec: dict, is_summary: bool) -> str:
    score = _summary_fact_priority(rec) if is_summary else _fact_clinical_priority(rec)
    if score >= 75:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _event_severity_label(rec: dict, is_summary: bool) -> str:
    score = _summary_fact_priority(rec) if is_summary else _fact_clinical_priority(rec)
    return "major" if score >= 60 else "minor"


def _event_concept_tokens(text: str) -> List[str]:
    lowered = (text or "").lower()
    lowered = re.sub(r"\b\d+(?:\.\d+)?\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9%/+\-]+", " ", lowered)
    toks: List[str] = []
    for tok in lowered.split():
        if len(tok) <= 2:
            continue
        if tok in EVENT_STOPWORDS:
            continue
        toks.append(tok)
    return toks


def _event_concept_key(rec: dict) -> str:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "")
    tokens = _event_concept_tokens(text)
    if not tokens:
        return re.sub(r"\s+", " ", text.lower()).strip()[:80]
    return " ".join(tokens[:5])


def _normalized_source_bucket(rec: dict) -> Optional[int]:
    src_ids = rec.get("source_sentence_ids") or []
    for raw in src_ids:
        if isinstance(raw, int):
            return raw
        s = str(raw).strip()
        if s.isdigit():
            return int(s)
        m = re.fullmatch(r"[sS](\d+)", s)
        if m:
            return int(m.group(1))
    return None


def _extract_event_slots(rec: dict) -> dict:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "").strip()
    lowered = text.lower()
    slots: Dict[str, Any] = {}

    laterality = None
    if "bilateral" in lowered:
        laterality = "bilateral"
    elif "right" in lowered:
        laterality = "right"
    elif "left" in lowered:
        laterality = "left"
    if laterality:
        slots["laterality"] = laterality

    status = str(rec.get("status") or "").strip().lower()
    if status:
        slots["assertion"] = status

    temporality = str(rec.get("temporality") or "").strip().lower()
    if temporality:
        slots["temporality"] = temporality

    value_matches = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|cm|mm|%|hours?|days?|weeks?|months?)\b", lowered)
    if value_matches:
        slots["values"] = value_matches[:3]

    body_sites = []
    for term in ["chest", "back", "groin", "abdomen", "ankle", "kidney", "ureter", "hand", "leg", "urine", "shoulder", "heart", "lung"]:
        if term in lowered:
            body_sites.append(term)
    if body_sites:
        slots["body_sites"] = body_sites[:3]

    actions = []
    for term in ["start", "stop", "continue", "hold", "increase", "decrease", "prescribe", "ordered", "order"]:
        if term in lowered:
            actions.append(term)
    if actions:
        slots["actions"] = actions[:3]

    plan_terms = []
    for term in [
        "follow up", "follow-up", "return", "call", "seek care", "lithotripsy",
        "bmp", "urinalysis", "urine culture", "echocardiogram", "lipid panel",
    ]:
        if term in lowered:
            plan_terms.append(term)
    if plan_terms:
        slots["plan_terms"] = plan_terms[:4]
    return slots


def _asgari_omission_family(rec: dict, event_type: str) -> str:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "").lower()
    category = str(rec.get("category") or "").lower()
    if event_type in {"MedicationPlan", "TestProcedurePlan", "FollowUpSafety"}:
        return "Information and Plan"
    if event_type in {"SymptomState", "FindingResult", "DiagnosisImpression"}:
        return "Current Issues"
    if event_type == "MedicationState":
        if any(k in text for k in ["increase", "decrease", "start", "stop", "hold", "continue", "dose"]):
            return "Information and Plan"
        return "PMFS"
    if event_type == "PMFSContext" or category in {"history", "demographic", "allergy"}:
        return "PMFS"
    return "Current Issues"


def _asgari_hallucination_families(rec: dict, event_type: str) -> List[str]:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "").lower()
    status = str(rec.get("status") or "").lower()
    families: List[str] = []
    if status == "negated" or " no " in f" {text} " or "denies" in text:
        families.append("Negation")
    if any(k in text for k in ["because", "due to", "caused by", "related to", "secondary to"]):
        families.append("Causality")
    if any(k in text for k in ["history", "previous", "prior", "family", "past", "ago", "today", "current"]):
        families.append("Contextual")
    if event_type in {"MedicationPlan", "TestProcedurePlan", "FollowUpSafety", "FindingResult", "DiagnosisImpression"}:
        families.append("Fabrication")
    if not families:
        families.append("Fabrication")
    return sorted(dict.fromkeys(families))


def _summary_group_signature(rec: dict, event_type: str, concept_key: str) -> Tuple[str, str, Optional[int]]:
    source_bucket = _normalized_source_bucket(rec)
    slots = _extract_event_slots(rec)
    primary_terms: List[str] = []
    for key in ["plan_terms", "actions", "body_sites", "values"]:
        values = slots.get(key)
        if isinstance(values, list):
            primary_terms.extend([str(v) for v in values[:2]])
    if not primary_terms:
        primary_terms = concept_key.split()[:3]
    normalized_terms = " ".join(sorted(dict.fromkeys(primary_terms))[:4])
    if event_type in {"MedicationPlan", "TestProcedurePlan", "FollowUpSafety", "FindingResult", "MedicationState"}:
        return (event_type, normalized_terms, source_bucket)
    if event_type == "SymptomState" and source_bucket is not None:
        return (event_type, normalized_terms, source_bucket)
    return (event_type, concept_key, source_bucket)


def build_event_records(
    fact_records: List[dict],
    *,
    is_summary: bool,
) -> Tuple[List[dict], Dict[int, int]]:
    # Build event groups deterministically from extracted propositions/facts.
    # This is a local consolidation step, not an additional LLM generation pass.
    if not fact_records:
        return [], {}

    grouped: Dict[Tuple[str, str, Optional[int]], List[int]] = {}
    for idx, rec in enumerate(fact_records):
        event_type = _event_type_from_record(rec)
        concept_key = _event_concept_key(rec)
        if is_summary:
            group_key = _summary_group_signature(rec, event_type, concept_key)
        else:
            group_key = (event_type, concept_key, None)
        grouped.setdefault(group_key, []).append(idx)

    events: List[dict] = []
    fact_to_event: Dict[int, int] = {}
    prefix = "SE" if is_summary else "TE"
    for event_idx, ((event_type, concept_key, source_bucket), member_ids) in enumerate(grouped.items()):
        best_idx = max(
            member_ids,
            key=lambda i: (_summary_fact_priority(fact_records[i]) if is_summary else _fact_clinical_priority(fact_records[i])),
        )
        best = fact_records[best_idx]
        event_slots = _extract_event_slots(best)
        event = {
            "event_id": f"{prefix}{event_idx}",
            "event_type": event_type,
            "clinical_priority": _event_priority_label(best, is_summary),
            "severity": _event_severity_label(best, is_summary),
            "event_anchor": str(best.get("fact_text") or best.get("proposition_text") or "").strip(),
            "status": best.get("status") or None,
            "temporality": best.get("temporality") or None,
            "speaker": best.get("speaker") or None,
            "source_sentence_ids": sorted({
                int(_normalized_source_bucket({"source_sentence_ids": [sid]}))
                for rec in [fact_records[i] for i in member_ids]
                for sid in (rec.get("source_sentence_ids") or [])
                if _normalized_source_bucket({"source_sentence_ids": [sid]}) is not None
            }) if is_summary else None,
            "linked_fact_ids": [f"S{i}" if is_summary else f"F{i}" for i in member_ids],
            "representative_fact_index": best_idx,
            "core_concepts": concept_key.split()[:5],
            "event_slots": event_slots,
            "asgari_alignment": {
                "omission_family": _asgari_omission_family(best, event_type),
                "hallucination_family_candidates": _asgari_hallucination_families(best, event_type),
            },
        }
        event = {k: v for k, v in event.items() if v not in (None, [], "")}
        events.append(event)
        for member_idx in member_ids:
            fact_to_event[member_idx] = event_idx
    return events, fact_to_event


def format_indexed_event_records(event_records: List[dict]) -> str:
    return "\n".join(f"{rec['event_id']}: {json.dumps(rec, ensure_ascii=False)}" for rec in event_records)


def parse_event_label_list(items: Any, prefix: str) -> List[int]:
    out: List[int] = []
    if not isinstance(items, list):
        items = [items]
    for x in items:
        if isinstance(x, int):
            out.append(x)
            continue
        s = str(x).strip()
        m = re.fullmatch(rf"{prefix}(\d+)", s, flags=re.IGNORECASE)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def normalize_detection_fact_records(obj: Any) -> list[dict]:
    if obj is None:
        return []
    if isinstance(obj, str):
        parsed = _try_parse_json_string(obj)
        if parsed is not obj:
            return normalize_detection_fact_records(parsed)
        lines = [ln.strip("- ").strip() for ln in obj.splitlines() if ln.strip()]
        return [{
            "fact_text": ln,
            "proposition_text": ln,
            "provenance": {"span_text": ""},
            "category": "",
            "speaker": "",
            "predicate": "",
            "status": "",
            "temporality": "",
            "value": None,
            "severity": None,
        } for ln in lines]
    if isinstance(obj, dict):
        if isinstance(obj.get("atomic_propositions"), list):
            out = []
            for item in obj["atomic_propositions"]:
                out.extend(normalize_detection_fact_records(item))
            return out
        rec = dict(obj)
        fact_text = rec.get("fact_text") or rec.get("proposition_text") or rec.get("text") or json.dumps(rec, ensure_ascii=False)
        rec["fact_text"] = str(fact_text).strip()
        rec["provenance"] = rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {"span_text": ""}
        return [rec] if rec["fact_text"] else []
    if isinstance(obj, list):
        out = []
        for x in obj:
            out.extend(normalize_detection_fact_records(x))
        return out
    return []


def _summary_fact_from_sentence(sentence: str, idx: int) -> dict:
    return {"fact_id": f"SF{idx+1}", "source_sentence_ids": [idx], "fact_text": sentence}


def normalize_summary_atomic_facts(obj: Any, summary_sentences: list[str]) -> list[dict]:
    if obj is None:
        return [_summary_fact_from_sentence(s, i) for i, s in enumerate(summary_sentences)]
    if isinstance(obj, str):
        parsed = _try_parse_json_string(obj)
        if parsed is not obj:
            return normalize_summary_atomic_facts(parsed, summary_sentences)
        return [_summary_fact_from_sentence(s, i) for i, s in enumerate(summary_sentences)]
    if isinstance(obj, dict):
        items = obj.get("summary_atomic_facts") if isinstance(obj.get("summary_atomic_facts"), list) else obj.get("atomic_propositions") if isinstance(obj.get("atomic_propositions"), list) else [obj]
    elif isinstance(obj, list):
        items = obj
    else:
        items = []
    out = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({"fact_id": f"SF{i+1}", "source_sentence_ids": [i], "fact_text": item})
            continue
        if not isinstance(item, dict):
            continue
        fact_text = item.get("fact_text") or item.get("proposition_text") or item.get("text")
        if not fact_text:
            continue
        source_sentence_ids = item.get("source_sentence_ids") or []
        norm_ids = []
        for x in source_sentence_ids:
            if isinstance(x, int):
                norm_ids.append(x)
            elif str(x).isdigit():
                norm_ids.append(int(x))
            else:
                m = re.fullmatch(r"[sS](\d+)", str(x).strip())
                if m:
                    norm_ids.append(int(m.group(1)))
        out.append({
            "fact_id": item.get("fact_id") or f"SF{i+1}",
            "source_sentence_ids": norm_ids,
            "fact_text": str(fact_text).strip(),
            "category": item.get("category") or "",
            "predicate": item.get("predicate") or "",
            "status": item.get("status") or "",
            "temporality": item.get("temporality") or "",
            "claim_type_tags": item.get("claim_type_tags") or [],
            "proposition_text": item.get("proposition_text") or str(fact_text).strip(),
        })
    return out or [_summary_fact_from_sentence(s, i) for i, s in enumerate(summary_sentences)]


def _fact_clinical_priority(rec: dict) -> int:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "")
    category = str(rec.get("category") or "").lower()
    predicate = str(rec.get("predicate") or "").lower()
    tags = {str(x).lower() for x in (rec.get("claim_type_tags") or [])}
    score = 0
    if category in {"diagnosis", "medicationplan", "testplan", "followupplan", "allergy"}:
        score += 60
    elif category in {"symptom", "finding", "chiefcomplaint"}:
        score += 40
    elif category == "history":
        score += 20
    if predicate in {"diagnoses", "prescribes", "orders", "recommends", "instructs", "plans"}:
        score += 25
    if {"safety_relevant", "medication", "plan", "diagnostic", "quantitative"} & tags:
        score += 25
    if _text_has_any(text, HIGH_SEVERITY_KEYWORDS):
        score += 25
    if _looks_numeric_or_result_fact(text):
        score += 10
    return score


def _summary_fact_priority(rec: dict) -> int:
    text = str(rec.get("fact_text") or rec.get("proposition_text") or "")
    category = str(rec.get("category") or "").lower()
    tags = {str(x).lower() for x in (rec.get("claim_type_tags") or [])}
    score = 0
    if category in {"diagnosis", "medicationplan", "testplan", "followupplan", "allergy"}:
        score += 70
    elif category in {"symptom", "finding", "chiefcomplaint"}:
        score += 45
    elif category == "history":
        score += 15
    if {"safety_relevant", "medication", "plan", "diagnostic", "quantitative"} & tags:
        score += 30
    if _text_has_any(text, HIGH_SEVERITY_KEYWORDS):
        score += 30
    if _looks_numeric_or_result_fact(text):
        score += 10
    if _text_has_any(text, LOW_VALUE_HALLUCINATION_PATTERNS):
        score -= 35
    return score


def _should_include_fact_in_detection_prompt(rec: dict) -> bool:
    fact_text = str(rec.get("fact_text", "") or "").strip()
    if not fact_text:
        return False
    predicate = str(rec.get("predicate", "") or "").strip().lower()
    status = str(rec.get("status", "") or "").strip().lower()
    category = str(rec.get("category", "") or "").strip().lower()
    lowered = fact_text.lower()
    if predicate in {"asks", "inquires", "clarifies", "confirms"} and status in {"", "uncertain"}:
        return False
    if any(x in lowered for x in ["can you", "could you", "would you", "let's talk", "any questions right now"]):
        return False
    return category in {"chiefcomplaint", "symptom", "diagnosis", "medicationplan", "testplan", "followupplan", "procedure", "finding", "allergy"} or True


def _should_include_fact_in_omission_prompt(rec: dict) -> bool:
    fact_text = str(rec.get("fact_text", "") or "").strip()
    if not fact_text:
        return False
    predicate = str(rec.get("predicate", "") or "").strip().lower()
    status = str(rec.get("status", "") or "").strip().lower()
    category = str(rec.get("category", "") or "").strip().lower()
    lowered = fact_text.lower()
    if predicate in {"asks", "inquires", "clarifies", "confirms"} and status in {"", "uncertain"}:
        return False
    if any(x in lowered for x in ["question", "asks whether", "asks if", "clarifies whether", "let's talk", "review of systems sheet"]):
        return False
    strong_categories = {"chiefcomplaint", "symptom", "diagnosis", "medicationplan", "followupplan", "procedure", "allergy"}
    if category in strong_categories:
        return True
    if category == "finding" and _looks_numeric_or_result_fact(fact_text):
        return True
    return _text_has_any(fact_text, HIGH_SEVERITY_KEYWORDS)


def _should_include_fact_in_omission_prompt_s_only(rec: dict) -> bool:
    fact_text = str(rec.get("fact_text", "") or rec.get("proposition_text", "") or "").strip()
    if not fact_text:
        return False
    predicate = str(rec.get("predicate", "") or "").strip().lower()
    status = str(rec.get("status", "") or "").strip().lower()
    category = str(rec.get("category", "") or "").strip().lower()
    speaker = str(rec.get("speaker", "") or "").strip().lower()
    lowered = fact_text.lower()

    if predicate in {"asks", "inquires", "clarifies", "confirms"} and status in {"", "uncertain"}:
        return False
    if any(x in lowered for x in ["question", "asks whether", "asks if", "clarifies whether", "let's talk", "review of systems sheet"]):
        return False
    if speaker in {"doctor", "clinician", "provider"}:
        return False

    if category in {"chiefcomplaint", "symptom", "history", "allergy", "demographic"}:
        return True
    if category == "medication" and speaker in {"patient", "", "unknown"}:
        return True
    if category == "diagnosis" and speaker in {"patient", "", "unknown"}:
        return True

    if category in {"finding", "result", "medicationplan", "testplan", "followupplan", "procedure", "assessment", "impression"}:
        return False

    subjective_markers = [
        "history of", "family history", "denies", "reports", "takes", "taking",
        "allergic", "allergy", "medication", "smokes", "drinks", "wine", "burger", "burgers",
    ]
    return speaker in {"patient", "", "unknown"} and any(marker in lowered for marker in subjective_markers)


def _should_include_fact_in_major_omission_verification(rec: dict) -> bool:
    fact_text = str(rec.get("fact_text", "") or rec.get("proposition_text", "") or "").strip()
    if not fact_text:
        return False
    predicate = str(rec.get("predicate", "") or "").strip().lower()
    status = str(rec.get("status", "") or "").strip().lower()
    lowered = fact_text.lower()
    if predicate in {"asks", "inquires", "clarifies", "confirms"} and status in {"", "uncertain"}:
        return False
    if any(x in lowered for x in ["question", "asks whether", "asks if", "clarifies whether", "let's talk", "review of systems sheet"]):
        return False
    return _fact_clinical_priority(rec) >= 40


def _event_rank_tuple(event: dict) -> Tuple[int, int, int]:
    priority_map = {"high": 3, "medium": 2, "low": 1}
    type_map = {
        "DiagnosisImpression": 7,
        "MedicationPlan": 6,
        "MedicationState": 6,
        "FindingResult": 5,
        "TestProcedurePlan": 4,
        "FollowUpSafety": 4,
        "SymptomState": 3,
        "PMFSContext": 1,
    }
    return (
        priority_map.get(str(event.get("clinical_priority") or "low").lower(), 1),
        1 if str(event.get("severity") or "minor").lower() == "major" else 0,
        type_map.get(str(event.get("event_type") or ""), 0),
    )


def _compact_fact_json_for_prompt(rec: dict) -> str:
    payload = {
        "prop_id": rec.get("prop_id") or None,
        "category": rec.get("category") or None,
        "speaker": rec.get("speaker") or None,
        "predicate": rec.get("predicate") or None,
        "status": rec.get("status") or None,
        "temporality": rec.get("temporality") or None,
        "value": rec.get("value") if rec.get("value") not in [""] else None,
        "severity": rec.get("severity") if rec.get("severity") not in [""] else None,
        "proposition_text": rec.get("fact_text") or None,
        "provenance": rec.get("provenance") or None,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return json.dumps(payload, ensure_ascii=False)


def format_indexed_sentences(sentences: list[str], prefix: str = "S") -> str:
    return "\n".join([f"{prefix}{i}: {s}" for i, s in enumerate(sentences)])


def format_indexed_fact_records_full(fact_records: list[dict], prefix: str = "F") -> str:
    return "\n".join(f"{prefix}{i}: {json.dumps(rec, ensure_ascii=False)}" for i, rec in enumerate(fact_records))


def format_indexed_summary_facts(summary_fact_records: list[dict]) -> str:
    lines = []
    for i, rec in enumerate(summary_fact_records):
        payload = {
            "fact_id": rec.get("fact_id") or f"SF{i+1}",
            "source_sentence_ids": rec.get("source_sentence_ids", []),
            "category": rec.get("category") or None,
            "predicate": rec.get("predicate") or None,
            "status": rec.get("status") or None,
            "temporality": rec.get("temporality") or None,
            "claim_type_tags": rec.get("claim_type_tags") or None,
            "fact_text": rec.get("fact_text", ""),
        }
        payload = {k: v for k, v in payload.items() if v not in (None, [], "")}
        lines.append(f"S{i}: {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def parse_index_label_list(items: Any, prefix: str) -> list[int]:
    out = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, int):
            out.append(x)
            continue
        s = str(x).strip()
        m = re.fullmatch(rf"{prefix}(\d+)", s, flags=re.IGNORECASE)
        if m:
            out.append(int(m.group(1)))
            continue
        if s.isdigit():
            out.append(int(s))
    return sorted(set(out))


def _normalize_reason_list(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, list):
        return [str(x).strip() for x in items if str(x).strip()]
    s = str(items).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return [s]


def _build_summary_fact_maps(summary_fact_records: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    outer_map, inner_map = {}, {}
    for i, rec in enumerate(summary_fact_records):
        outer_map[f"S{i}".upper()] = i
        fact_id = str(rec.get("fact_id", "") or "").strip()
        if fact_id:
            inner_map[fact_id.upper()] = i
    return outer_map, inner_map


def _extract_summary_fact_indices_any(value: Any, summary_fact_records: list[dict]) -> list[int]:
    outer_map, inner_map = _build_summary_fact_maps(summary_fact_records)
    out = []
    items = value if isinstance(value, list) else [value]
    for x in items:
        s = str(x).strip().upper()
        if not s:
            continue
        if s in outer_map:
            out.append(outer_map[s])
            continue
        if s in inner_map:
            out.append(inner_map[s])
            continue
        m = re.fullmatch(r"S(\d+)", s)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(summary_fact_records):
                out.append(idx)
            continue
        m = re.fullmatch(r"SF(\d+)", s)
        if m:
            key = f"SF{int(m.group(1))}"
            if key in inner_map:
                out.append(inner_map[key])
    return sorted(set(out))


def _recover_summary_fact_indices_from_reasons_robust(reasons: list[str], summary_fact_records: list[dict]) -> list[int]:
    outer_map, inner_map = _build_summary_fact_maps(summary_fact_records)
    out = []
    for reason in reasons or []:
        text = str(reason)
        for m in re.finditer(r"\bS(\d+)\b", text, flags=re.IGNORECASE):
            key = f"S{int(m.group(1))}".upper()
            if key in outer_map:
                out.append(outer_map[key])
        for m in re.finditer(r"\bSF(\d+)\b", text, flags=re.IGNORECASE):
            key = f"SF{int(m.group(1))}".upper()
            if key in inner_map:
                out.append(inner_map[key])
    return sorted(set(out))


def _extract_index_from_reason(reason: str, prefix: str) -> Optional[int]:
    m = re.search(rf"\b{prefix}(\d+)\b", str(reason), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _recover_fact_indices_from_reasons(reasons: list[str], fact_records: list[dict]) -> list[int]:
    out = []
    for reason in reasons or []:
        idx = _extract_index_from_reason(reason, "F")
        if idx is not None and 0 <= idx < len(fact_records):
            out.append(idx)
    return sorted(set(out))


def _align_reasons_to_ids(reasons: list[str], ids: list[int], prefix: str) -> list[str]:
    if not ids:
        return []
    reasons = _normalize_reason_list(reasons)
    bucket = {idx: [] for idx in ids}
    leftovers = []
    for reason in reasons:
        idx = _extract_index_from_reason(reason, prefix)
        if idx is not None and idx in bucket:
            bucket[idx].append(str(reason).strip())
        else:
            leftovers.append(str(reason).strip())
    aligned = []
    leftover_iter = iter(leftovers)
    for idx in ids:
        if bucket[idx]:
            aligned.append(bucket[idx][0])
        else:
            try:
                aligned.append(next(leftover_iter))
            except StopIteration:
                aligned.append(f"{prefix}{idx}: clinically important unsupported/missing event detected.")
    return aligned


def extract_summary_atomic_facts(method_key: str, full_summary: str, *, model: str = DEFAULT_MODEL) -> dict:
    user_prompt, summary_sentences = build_summary_extract_prompt(method_key, full_summary)
    try:
        text = llm_generate("extraction", user_prompt, max_tokens=8192)
        obj = safe_json_extract(text)
    except Exception:
        obj = {"summary_atomic_facts": [_summary_fact_from_sentence(s, i) for i, s in enumerate(summary_sentences)]}
    return obj


def _safe_json_detect(user_prompt: str) -> dict:
    try:
        text = llm_generate("detection", user_prompt, max_tokens=8192)
        out = safe_json_extract(text)
        out.setdefault("detection_status", "ok")
        out.setdefault("detection_error", "")
        return out
    except Exception as exc:
        return {
            "unsupported_sentence_ids": [],
            "unsupported_sentences": [],
            "omitted_event_ids_top1": [],
            "omitted_event_ids_top3": [],
            "omitted_event_verifications": [],
            "omitted_event_verdicts": [],
            "omitted_event_reason_tags": [],
            "omitted_fact_ids": [],
            "omitted_facts": [],
            "hallucination_reasons": [],
            "omission_reasons": [],
            "total_count": 0,
            "total_hallucination_count": 0,
            "total_omission_count": 0,
            "detection_status": "error",
            "detection_error": str(exc),
        }


def _empty_hallucination_detection_result() -> dict:
    return {
        "unsupported_summary_event_ids": [],
        "unsupported_summary_events": [],
        "unsupported_summary_fact_ids": [],
        "unsupported_summary_facts": [],
        "unsupported_sentence_ids": [],
        "unsupported_sentences": [],
        "hallucination_reasons": [],
        "total_hallucination_count": 0,
        "detection_status": "skipped",
        "detection_error": "",
    }


def _empty_omission_detection_result() -> dict:
    return {
        "omitted_event_ids": [],
        "omitted_event_ids_top1": [],
        "omitted_event_ids_top3": [],
        "omitted_events": [],
        "omitted_event_verifications": [],
        "omitted_event_verdicts": [],
        "omitted_event_reason_tags": [],
        "omitted_fact_ids": [],
        "omitted_facts": [],
        "omission_reasons": [],
        "total_omission_count": 0,
        "detection_status": "skipped",
        "detection_error": "",
    }


def _normalize_selected_axes(selected_axes: Optional[Iterable[str]]) -> List[str]:
    if selected_axes is None:
        return list(DETECTION_AXES)
    normalized: List[str] = []
    for axis in selected_axes:
        axis_norm = str(axis).strip().lower()
        if not axis_norm:
            continue
        if axis_norm not in DETECTION_AXES:
            raise ValueError(f"Unsupported detection axis: {axis}")
        if axis_norm not in normalized:
            normalized.append(axis_norm)
    return normalized or list(DETECTION_AXES)


def detect_hallucinations_stress_symmetric(transcript_ciu: Any, summary_atomic_facts: Any, full_summary: str, *, method_key: str, model: str = DEFAULT_MODEL) -> dict:
    transcript_fact_records = normalize_detection_fact_records(transcript_ciu)
    transcript_prompt_records = sorted([rec for rec in transcript_fact_records if _should_include_fact_in_detection_prompt(rec)], key=_fact_clinical_priority, reverse=True)
    summary_sentences = split_summary_body_sentences(full_summary)
    summary_fact_records = normalize_summary_atomic_facts(summary_atomic_facts, summary_sentences)
    transcript_event_records, _ = build_event_records(transcript_prompt_records, is_summary=False)
    summary_event_records, summary_fact_to_event = build_event_records(summary_fact_records, is_summary=True)
    prompt = METHODS[method_key].get("detect_prompt_hallucination_stress", METHODS[method_key]["detect_prompt_stress"]).format(
        indexed_facts=format_indexed_event_records(transcript_event_records),
        indexed_summary_facts=format_indexed_event_records(summary_event_records),
    )
    raw = _safe_json_detect(prompt)
    reasons = _normalize_reason_list(raw.get("hallucination_reasons", raw.get("reasons", [])))
    unsupported_event_ids = parse_event_label_list(raw.get("unsupported_summary_event_ids", []), prefix="SE")
    if not unsupported_event_ids:
        unsupported_event_ids = parse_event_label_list(raw.get("unsupported_summary_fact_ids", []), prefix="SE")
    if not unsupported_event_ids:
        unsupported_fact_ids = _extract_summary_fact_indices_any(raw.get("unsupported_summary_fact_ids", []), summary_fact_records)
        unsupported_event_ids = sorted({
            event_idx
            for fact_idx, event_idx in summary_fact_to_event.items()
            if fact_idx in unsupported_fact_ids
        })
    if not unsupported_event_ids and reasons:
        for reason in reasons:
            idx = _extract_index_from_reason(reason, "SE")
            if idx is not None and 0 <= idx < len(summary_event_records):
                unsupported_event_ids.append(idx)
        unsupported_event_ids = sorted(set(unsupported_event_ids))
    unsupported_event_ids = [i for i in unsupported_event_ids if 0 <= i < len(summary_event_records)]

    representative_fact_ids: List[int] = []
    representative_sentence_ids: List[int] = []
    for event_idx in unsupported_event_ids:
        event = summary_event_records[event_idx]
        fact_idx = int(event.get("representative_fact_index", -1))
        if 0 <= fact_idx < len(summary_fact_records):
            representative_fact_ids.append(fact_idx)
        for sent_idx in event.get("source_sentence_ids", []) or []:
            if 0 <= int(sent_idx) < len(summary_sentences):
                representative_sentence_ids.append(int(sent_idx))
    representative_fact_ids = sorted(dict.fromkeys(representative_fact_ids))
    representative_sentence_ids = sorted(dict.fromkeys(representative_sentence_ids))
    return {
        "unsupported_summary_event_ids": unsupported_event_ids,
        "unsupported_summary_events": [summary_event_records[i].get("event_anchor", "") for i in unsupported_event_ids],
        "unsupported_summary_fact_ids": representative_fact_ids,
        "unsupported_summary_facts": [summary_fact_records[i].get("fact_text", "") for i in representative_fact_ids],
        "unsupported_sentence_ids": representative_sentence_ids,
        "unsupported_sentences": [summary_sentences[i] for i in representative_sentence_ids if 0 <= i < len(summary_sentences)],
        "hallucination_reasons": _align_reasons_to_ids(reasons, unsupported_event_ids, "SE"),
        "total_hallucination_count": len(unsupported_event_ids),
        "detection_status": raw.get("detection_status", "ok"),
        "detection_error": raw.get("detection_error", ""),
    }


def detect_omissions_stress_asymmetric(transcript_ciu: Any, full_summary: str, *, method_key: str, model: str = DEFAULT_MODEL) -> dict:
    fact_records = normalize_detection_fact_records(transcript_ciu)
    prompt_fact_records = [rec for rec in fact_records if _should_include_fact_in_major_omission_verification(rec)]
    prompt_fact_records = sorted(prompt_fact_records, key=_fact_clinical_priority, reverse=True)
    transcript_event_records, _ = build_event_records(prompt_fact_records, is_summary=False)
    summary_sentences = split_summary_body_sentences(full_summary)
    prompt = METHODS[method_key].get("detect_prompt_omission_stress", METHODS[method_key]["detect_prompt_stress"]).format(
        indexed_facts=format_indexed_event_records(transcript_event_records),
        indexed_summary_sentences=format_indexed_sentences(summary_sentences, prefix="S"),
    )
    raw = _safe_json_detect(prompt)
    reasons = _normalize_reason_list(raw.get("omission_reasons", []))
    verifications = raw.get("event_verifications", [])
    if not isinstance(verifications, list):
        verifications = []
    verification_map: Dict[int, dict] = {}
    for item in verifications:
        if not isinstance(item, dict):
            continue
        for idx in parse_event_label_list(item.get("event_id"), prefix="TE"):
            if 0 <= idx < len(transcript_event_records):
                verification_map[idx] = item

    ranked_ids = parse_event_label_list(raw.get("ranked_missing_major_event_ids", []), prefix="TE")
    if not ranked_ids:
        derived_ids: List[int] = []
        for item in verifications:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict != "missingmajor":
                continue
            idxs = parse_event_label_list(item.get("event_id"), prefix="TE")
            derived_ids.extend(idxs)
        ranked_ids = sorted(set(derived_ids))
    if not ranked_ids and reasons:
        for reason in reasons:
            idx = _extract_index_from_reason(reason, "TE")
            if idx is not None and 0 <= idx < len(transcript_event_records):
                ranked_ids.append(idx)
        ranked_ids = sorted(set(ranked_ids))

    ranked_ids = [i for i in ranked_ids if 0 <= i < len(transcript_event_records)]
    ranked_ids = sorted(
        ranked_ids,
        key=lambda i: _event_rank_tuple(transcript_event_records[i]),
        reverse=True,
    )
    omitted_event_ids = ranked_ids

    omitted_ids: List[int] = []
    for event_idx in omitted_event_ids:
        fact_idx = int(transcript_event_records[event_idx].get("representative_fact_index", -1))
        if 0 <= fact_idx < len(prompt_fact_records):
            omitted_ids.append(fact_idx)
    omitted_ids = sorted(dict.fromkeys(omitted_ids))

    omission_events: List[dict] = []
    omitted_event_verdicts: List[str] = []
    omitted_event_reason_tags: List[str] = []
    for event_idx in omitted_event_ids:
        event = transcript_event_records[event_idx]
        verification = verification_map.get(event_idx, {})
        verdict = str(verification.get("verdict", "MissingMajor") or "MissingMajor").strip()
        reason_tag = str(verification.get("reason_tag", "") or "").strip()
        reason = str(verification.get("reason", "") or "").strip()
        omitted_event_verdicts.append(verdict)
        omitted_event_reason_tags.append(reason_tag)
        omission_events.append({
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "severity": event.get("severity", "minor"),
            "clinical_priority": event.get("clinical_priority", "low"),
            "omission_verdict": verdict,
            "omission_reason_tag": reason_tag,
            "omission_reason": reason,
            "event_anchor": event.get("event_anchor", ""),
            "linked_fact_ids": event.get("linked_fact_ids", []),
        })
    top1_ids = omitted_event_ids[:1]
    top3_ids = omitted_event_ids[:3]
    return {
        "omitted_event_ids": omitted_event_ids,
        "omitted_event_ids_top1": top1_ids,
        "omitted_event_ids_top3": top3_ids,
        "omitted_events": omission_events,
        "omitted_event_verifications": verifications,
        "omitted_event_verdicts": omitted_event_verdicts,
        "omitted_event_reason_tags": omitted_event_reason_tags,
        "omitted_fact_ids": omitted_ids,
        "omitted_facts": [prompt_fact_records[i].get("fact_text") or prompt_fact_records[i].get("proposition_text") or json.dumps(prompt_fact_records[i], ensure_ascii=False) for i in omitted_ids],
        "omission_reasons": _align_reasons_to_ids(reasons, omitted_event_ids, "TE"),
        "total_omission_count": len(omitted_event_ids),
        "detection_status": raw.get("detection_status", "ok"),
        "detection_error": raw.get("detection_error", ""),
    }


def detect_hallucinations_and_omissions_v1(
    transcript_ciu: Any,
    full_summary: str,
    *,
    summary_atomic_facts: Any = None,
    method_key: str = "A",
    model: str = DEFAULT_MODEL,
    selected_axes: Optional[Iterable[str]] = None,
) -> dict:
    axes = _normalize_selected_axes(selected_axes)
    hallucination_result = (
        detect_hallucinations_stress_symmetric(transcript_ciu, summary_atomic_facts, full_summary, method_key=method_key, model=model)
        if "hallucination" in axes else _empty_hallucination_detection_result()
    )
    omission_result = (
        detect_omissions_stress_asymmetric(transcript_ciu, full_summary, method_key=method_key, model=model)
        if "omission" in axes else _empty_omission_detection_result()
    )
    return {
        **hallucination_result,
        **omission_result,
        "total_hallucination_count": hallucination_result.get("total_hallucination_count", 0),
        "total_omission_count": omission_result.get("total_omission_count", 0),
        "selected_axes": axes,
    }


# ============================================================
# Dataset / runner
# ============================================================

def load_dataset(input_path: str) -> pd.DataFrame:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".jsonl":
        records = []
        with open(input_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError(f"Loaded empty DataFrame from JSONL: {input_path}")
        if "file" not in df.columns:
            for cand in ["case_id", "encounter_id", "id", "uid"]:
                if cand in df.columns:
                    df["file"] = df[cand]
                    break
        if "summary" not in df.columns:
            for cand in ["summary_draft", "summary_gt_note", "summary_asgari_error"]:
                if cand in df.columns:
                    df["summary"] = df[cand]
                    break
        return df
    return pd.read_csv(input_path)


def _resolve_summary(row: pd.Series, dataset_id: str) -> str:
    dataset_id = dataset_id.lower()
    if dataset_id == "stress":
        for col in ["summary_draft", "summary", "summary_asgari_error", "summary_draft_original", "summary_gt_note"]:
            if col in row and pd.notna(row[col]):
                return str(row[col])
    for col in ["summary", "summary_draft", "summary_asgari_error", "summary_draft_original", "summary_gt_note"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ""


def stringify_extraction_output(obj: Any) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2)


def load_existing_detection_output(output_path: str) -> Optional[pd.DataFrame]:
    p = Path(output_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty or "file" not in df.columns:
        return None
    return df


def _is_nonempty_value(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and pd.isna(x):
        return False
    s = str(x).strip()
    return s not in {"", "nan", "None", "null"}


def save_results(results: list[dict], output_path: str) -> None:
    df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_audit(
    input_path: str,
    target_cases: Optional[list[str]],
    dataset_id: str,
    method_key: str,
    output_path: str,
    kiwi_mapping_path: Optional[str] = None,
    existing_facts_path: Optional[str] = None,
    model: Optional[str] = None,
    selected_axes: Optional[Iterable[str]] = None,
) -> None:
    model = model or DEFAULT_MODEL
    axes = _normalize_selected_axes(selected_axes)
    data = load_dataset(input_path)
    existing_df = load_existing_detection_output(output_path)
    done_files = set(existing_df["file"].astype(str).tolist()) if existing_df is not None else set()
    results = [] if existing_df is None else existing_df.to_dict("records")

    for _, row in data.iterrows():
        case_id = str(row.get("case_id") or row.get("file") or "")
        if target_cases and case_id not in target_cases:
            continue
        if case_id in done_files:
            print(f"[{dataset_id}-{method_key}] {case_id} skipped (already exists in output CSV).")
            continue

        transcript = str(row.get("transcript") or "")
        full_summary = _resolve_summary(row, dataset_id)

        kiwi_payload = None
        if kiwi_mapping_path and method_key in {"D", "E"}:
            kiwi_json = load_kiwi_mapping_for_case(kiwi_mapping_path, case_id)
            kiwi_payload = get_case_payload_from_kiwi_mapping(kiwi_json, case_id)
            if kiwi_payload is None:
                print(f"[WARN] Could not find payload for case_id={case_id} inside kiwi json")

        print(f"[{dataset_id}-{method_key}] {case_id} extraction...")
        extraction_started = time.perf_counter()
        extraction_time_parts: List[float] = []
        cap_chunk_count = 1
        if method_key == "A":
            part_started = time.perf_counter()
            extracted = extract_method_a_bn_atomic_facts(transcript, model=model)
            extraction_time_parts = [round(time.perf_counter() - part_started, 3)]
        elif method_key == "B":
            part_started = time.perf_counter()
            extracted = extract_method_b_verifact_atomic_claims(transcript, model=model)
            extraction_time_parts = [round(time.perf_counter() - part_started, 3)]
        elif method_key == "C":
            extracted, extraction_time_parts, cap_chunk_count = extract_cap_with_chunking(
                "C",
                transcript,
                model=model,
            )
        elif method_key == "D":
            extracted, extraction_time_parts, cap_chunk_count = extract_cap_with_chunking(
                "D",
                transcript,
                entities=(kiwi_payload or {}).get("entities", []),
                relations=(kiwi_payload or {}).get("relations", []),
                model=model,
            )
        elif method_key == "E":
            extracted, extraction_time_parts, cap_chunk_count = extract_cap_with_chunking(
                "E",
                transcript,
                entities=(kiwi_payload or {}).get("entities", []),
                relations=(kiwi_payload or {}).get("relations", []),
                model=model,
            )
        else:
            raise ValueError(f"Unsupported method key: {method_key}")
        extraction_time_sec = round(time.perf_counter() - extraction_started, 3)

        detection_started = time.perf_counter()
        summary_atomic: dict | None = None
        if "hallucination" in axes:
            print(f"[{dataset_id}-{method_key}] {case_id} summary atomic extraction...")
            summary_atomic = extract_summary_atomic_facts(method_key, full_summary, model=model)

        print(f"[{dataset_id}-{method_key}] {case_id} detection ({', '.join(axes)})...")
        detection = detect_hallucinations_and_omissions_v1(
            extracted,
            full_summary,
            summary_atomic_facts=summary_atomic,
            method_key=method_key,
            model=model,
            selected_axes=axes,
        )
        detection_time_sec = round(time.perf_counter() - detection_started, 3)

        result_row = {
            "file": case_id,
            "dataset_id": dataset_id,
            "method_key": method_key,
            "selected_axes": json.dumps(axes, ensure_ascii=False),
            "extraction_time_sec": extraction_time_sec,
            "extraction_time_sec_parts": json.dumps(extraction_time_parts, ensure_ascii=False),
            "cap_chunk_count": cap_chunk_count,
            "detection_time_sec": detection_time_sec,
            "transcript_facts": stringify_extraction_output(extracted),
            "summary_atomic_facts": json.dumps(summary_atomic or {}, ensure_ascii=False),
            "unsupported_summary_event_ids": json.dumps(detection.get("unsupported_summary_event_ids", []), ensure_ascii=False),
            "unsupported_summary_events": json.dumps(detection.get("unsupported_summary_events", []), ensure_ascii=False),
            "unsupported_summary_fact_ids": json.dumps(detection.get("unsupported_summary_fact_ids", []), ensure_ascii=False),
            "unsupported_sentences": json.dumps(detection.get("unsupported_sentences", []), ensure_ascii=False),
            "hallucination_reasons": json.dumps(detection.get("hallucination_reasons", []), ensure_ascii=False),
            "total_hallucination_count": detection.get("total_hallucination_count", 0),
            "omitted_event_ids": json.dumps(detection.get("omitted_event_ids", []), ensure_ascii=False),
            "omitted_event_ids_top1": json.dumps(detection.get("omitted_event_ids_top1", []), ensure_ascii=False),
            "omitted_event_ids_top3": json.dumps(detection.get("omitted_event_ids_top3", []), ensure_ascii=False),
            "omitted_events": json.dumps(detection.get("omitted_events", []), ensure_ascii=False),
            "omitted_event_verifications": json.dumps(detection.get("omitted_event_verifications", []), ensure_ascii=False),
            "omitted_event_verdicts": json.dumps(detection.get("omitted_event_verdicts", []), ensure_ascii=False),
            "omitted_event_reason_tags": json.dumps(detection.get("omitted_event_reason_tags", []), ensure_ascii=False),
            "omitted_fact_ids": json.dumps(detection.get("omitted_fact_ids", []), ensure_ascii=False),
            "omitted_facts": json.dumps(detection.get("omitted_facts", []), ensure_ascii=False),
            "omission_reasons": json.dumps(detection.get("omission_reasons", []), ensure_ascii=False),
            "total_omission_count": detection.get("total_omission_count", 0),
            "detection_status": detection.get("detection_status", "ok"),
            "detection_error": detection.get("detection_error", ""),
        }
        results.append(result_row)
        save_results(results, output_path)
        print(f"[{dataset_id}-{method_key}] {case_id} done. h={result_row['total_hallucination_count']} o={result_row['total_omission_count']}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--method_key", required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--kiwi_mapping_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--target_cases", nargs="*", default=None)
    parser.add_argument("--axes", nargs="+", choices=list(DETECTION_AXES), default=list(DETECTION_AXES))
    args = parser.parse_args()

    run_audit(
        input_path=args.input_path,
        target_cases=args.target_cases,
        dataset_id=args.dataset_id,
        method_key=args.method_key,
        output_path=args.output_path,
        kiwi_mapping_path=args.kiwi_mapping_path,
        model=args.model,
        selected_axes=args.axes,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc()
        raise
