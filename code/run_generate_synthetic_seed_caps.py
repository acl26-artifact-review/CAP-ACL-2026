from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_template_rendering_experiments as base


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = BASE_DIR / "outputs_legacy" / "shared" / "run_aci_all_v1_realistic" / "cases.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "internal_benchmark" / "generated_seed_caps"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "human_seed_caps_ab_gpt51_30.jsonl"


TURN_MARKER_RE = re.compile(r"\[(doctor|patient)\]", flags=re.IGNORECASE)

ALLOWED_CATEGORIES = [
    "Demographics",
    "Chief Complaint",
    "Past Medical History",
    "Social History",
    "Lifestyle",
    "Symptom",
    "Review of Systems",
    "Finding",
    "Physical Exam",
    "Test Result",
    "Assessment",
    "Procedure",
    "Medication",
    "MedicationPlan",
    "Plan",
    "Education",
    "Counseling",
]

CATEGORY_ALIASES = {
    "demographic": "Demographics",
    "demographics": "Demographics",
    "chief complaint": "Chief Complaint",
    "chief_complaint": "Chief Complaint",
    "past medical history": "Past Medical History",
    "pastmedicalhistory": "Past Medical History",
    "social history": "Social History",
    "socialhistory": "Social History",
    "lifestyle": "Lifestyle",
    "symptom": "Symptom",
    "review of systems": "Review of Systems",
    "reviewofsystems": "Review of Systems",
    "ros": "Review of Systems",
    "finding": "Finding",
    "physical exam": "Physical Exam",
    "physicalexam": "Physical Exam",
    "test result": "Test Result",
    "testresult": "Test Result",
    "assessment": "Assessment",
    "procedure": "Procedure",
    "medication": "Medication",
    "medication plan": "MedicationPlan",
    "medicationplan": "MedicationPlan",
    "plan": "Plan",
    "education": "Education",
    "counseling": "Counseling",
}

ALLOWED_SPEAKERS = {"patient", "clinician", "unknown"}
TEMPORALITY_MAP = {
    "past": "past",
    "current": "current",
    "future": "future",
    "past_to_current": "past_to_current",
    "past~current": "past_to_current",
    "past-current": "past_to_current",
    "ongoing": "past_to_current",
    "unknown": "unknown",
}
CERTAINTY_MAP = {
    "high": "high",
    "medium": "medium",
    "low": "low",
}


A_STYLE_INSTRUCTION = """
Annotator Style A (high precision, compact):
- Keep a distilled, high-salience CAP set.
- Prefer clinically central facts and decisions.
- Avoid over-fragmentation of tiny sub-facts.
- Include key negatives only when clinically relevant to current assessment/plan.
- Target roughly 25-45 CAPs depending on visit complexity.
""".strip()

B_STYLE_INSTRUCTION = """
Annotator Style B (higher recall, more granular):
- Keep all core facts from Style A, plus additional clinically useful detail.
- Allow finer granularity for symptom course, relevant context, and plan specifics.
- Include more supporting negatives and management details when they impact interpretation.
- Still avoid trivial conversational noise.
- Target roughly 35-65 CAPs depending on visit complexity.
""".strip()


@dataclass
class GenerationTask:
    case_id: str
    transcript: str
    annotator: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic A/B style human-seed CAP annotations using GPT-5.1."
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--task-workers", type=int, default=4)
    parser.add_argument("--max-transcript-chars", type=int, default=120000)
    parser.add_argument("--max-caps-a", type=int, default=45)
    parser.add_argument("--max-caps-b", type=int, default=65)
    parser.add_argument("--min-caps-a", type=int, default=20)
    parser.add_argument("--min-caps-b", type=int, default=30)
    parser.add_argument("--max-retries-on-thin-output", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--write-case-files", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def infer_base_url(model: str, explicit_base_url: Optional[str]) -> str:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")
    model_name = base.safe_text(model).strip().lower()
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return "https://api.openai.com/v1"
    return base.infer_api_base_url(None)


def infer_api_key(model: str, explicit_key: Optional[str]) -> str:
    if explicit_key is not None:
        return explicit_key
    model_name = base.safe_text(model).strip().lower()
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
        return os.getenv("OPENAI_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", os.getenv("RUNPOD_API_KEY", ""))


def split_transcript_into_turns(transcript: str) -> List[Tuple[int, str, str]]:
    text = base.safe_text(transcript).strip()
    if not text:
        return []

    matches = list(TURN_MARKER_RE.finditer(text))
    if not matches:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return [(idx, "unknown", line) for idx, line in enumerate(lines)]

    turns: List[Tuple[int, str, str]] = []
    for idx, match in enumerate(matches):
        speaker = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = re.sub(r"\s+", " ", text[start:end]).strip()
        if not content:
            continue
        turns.append((len(turns), speaker, content))
    return turns


def build_turn_block(turns: Sequence[Tuple[int, str, str]], max_chars: int) -> str:
    lines: List[str] = []
    total_chars = 0
    for turn_id, speaker, content in turns:
        line = f"{turn_id}: [{speaker}] {content}"
        line_len = len(line) + 1
        if lines and total_chars + line_len > max_chars:
            break
        lines.append(line)
        total_chars += line_len
    return "\n".join(lines)


def synthetic_seed_schema(max_caps: int, min_caps: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "seed_gold_caps": {
                "type": "array",
                "minItems": max(0, min_caps),
                "maxItems": max_caps,
                "items": {
                    "type": "object",
                    "properties": {
                        "gold_id": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "fact_text": {"type": "string"},
                        "category": {"type": "string"},
                        "speaker": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "temporality": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "certainty": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        "provenance_sentence": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]},
                    },
                    "required": ["fact_text", "category", "provenance_sentence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["seed_gold_caps"],
        "additionalProperties": False,
    }


def normalize_category(value: Any) -> str:
    raw = base.normalize_text(base.safe_text(value))
    if not raw:
        return "Finding"
    if raw in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[raw]
    # fallback: title-ish normalization
    candidate = " ".join(token.capitalize() for token in raw.split())
    if candidate in ALLOWED_CATEGORIES:
        return candidate
    return "Finding"


def normalize_speaker(value: Any) -> str:
    raw = base.normalize_text(base.safe_text(value))
    if raw in ALLOWED_SPEAKERS:
        return raw
    if raw in {"doctor", "provider", "clinician", "physician"}:
        return "clinician"
    if raw in {"patient", "pt"}:
        return "patient"
    return "unknown"


def normalize_temporality(value: Any) -> str:
    raw = base.normalize_text(base.safe_text(value))
    return TEMPORALITY_MAP.get(raw, "unknown")


def normalize_certainty(value: Any) -> str:
    raw = base.normalize_text(base.safe_text(value))
    return CERTAINTY_MAP.get(raw, "medium")


def normalize_provenance_sentence(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(int(value))
    if isinstance(value, list):
        parts = [normalize_provenance_sentence(x) for x in value]
        parts = [p for p in parts if p]
        return ",".join(parts)
    text = base.safe_text(value).strip()
    if not text:
        return ""
    text = text.replace("–", "-").replace("—", "-").replace("~", "-")
    text = text.replace(" to ", "-")
    # Keep only simple turn-id span patterns and separators.
    cleaned = re.sub(r"[^0-9,\-\s]", "", text)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned[:64]


def normalize_seed_caps(raw_obj: Dict[str, Any], *, max_caps: int) -> List[Dict[str, Any]]:
    raw_caps = raw_obj.get("seed_gold_caps")
    if not isinstance(raw_caps, list):
        return []
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for item in raw_caps:
        if not isinstance(item, dict):
            continue
        fact_text = base.safe_text(item.get("fact_text")).strip()
        if not fact_text:
            continue
        category = normalize_category(item.get("category"))
        speaker = normalize_speaker(item.get("speaker"))
        temporality = normalize_temporality(item.get("temporality"))
        certainty = normalize_certainty(item.get("certainty"))
        provenance_sentence = normalize_provenance_sentence(item.get("provenance_sentence"))
        dedup_key = (base.normalize_text(fact_text), category, provenance_sentence)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        out.append(
            {
                "gold_id": "",
                "fact_text": fact_text,
                "category": category,
                "speaker": speaker,
                "temporality": temporality,
                "certainty": certainty,
                "provenance_sentence": provenance_sentence,
            }
        )
        if len(out) >= max_caps:
            break
    for idx, cap in enumerate(out, start=1):
        cap["gold_id"] = f"G{idx}"
    return out


def build_prompt(
    case_id: str,
    annotator: str,
    turn_block: str,
    *,
    max_caps: int,
    min_caps: int,
    retry_hint: bool = False,
) -> str:
    style_block = A_STYLE_INSTRUCTION if annotator == "A" else B_STYLE_INSTRUCTION
    categories = ", ".join(ALLOWED_CATEGORIES)
    retry_block = ""
    if retry_hint:
        retry_block = (
            "Important retry instruction:\n"
            f"- Previous output was too sparse. Return at least {min_caps} seed_gold_caps.\n"
            "- Do not return an empty array.\n"
        )
    return f"""
You are creating internal synthetic human-seed CAP annotations for one clinical dialogue.
Return JSON only.

Case ID: {case_id}
Annotator: {annotator}

Task:
- Extract clinically meaningful atomic facts from the transcript turns.
- Use only explicit information from the transcript.
- Keep each fact atomic (one proposition per fact).
- Preserve key negations, temporality, uncertainty, and plan intent.
- Keep provenance_sentence grounded to turn ids shown below (e.g., "7", "12-13", "12,15").

{style_block}
{retry_block}

Output requirements:
- Return a JSON object with key "seed_gold_caps".
- Each item must contain:
  - gold_id (string or null)
  - fact_text (string)
  - category (one of: {categories})
  - speaker (patient | clinician | unknown)
  - temporality (past | current | future | past_to_current | unknown)
  - certainty (high | medium | low)
  - provenance_sentence (turn id / span string)
- Max number of CAPs: {max_caps}
- Minimum number of CAPs: {min_caps}
- Do not include unsupported assumptions or outside knowledge.

Transcript turns:
{turn_block}
""".strip()


def parse_llm_json(
    client: base.OpenAICompatClient,
    *,
    model: str,
    prompt: str,
    schema_obj: Dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    raw = base.call_llm(
        client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        force_json=True,
        json_schema_obj=schema_obj,
        prefer_json_object=True,
    )
    try:
        return base.safe_json_extract(raw)
    except Exception:
        return base.repair_json_via_llm(
            client,
            model=model,
            raw_text=raw,
            schema_obj=schema_obj,
            max_tokens=min(max_tokens, 2500),
        )


def generate_one(
    task: GenerationTask,
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
    max_transcript_chars: int,
    max_caps_a: int,
    max_caps_b: int,
    min_caps_a: int,
    min_caps_b: int,
    max_retries_on_thin_output: int,
) -> Dict[str, Any]:
    turns = split_transcript_into_turns(task.transcript)
    turn_block = build_turn_block(turns, max_chars=max_transcript_chars)
    max_caps = max_caps_a if task.annotator == "A" else max_caps_b
    min_caps = min_caps_a if task.annotator == "A" else min_caps_b
    client = base.OpenAICompatClient(base_url=base_url, api_key=api_key, timeout=timeout)
    caps: List[Dict[str, Any]] = []
    attempts_used = 0
    total_attempts = max(1, max_retries_on_thin_output + 1)
    for attempt_idx in range(total_attempts):
        attempts_used = attempt_idx + 1
        retry_hint = attempt_idx > 0
        prompt = build_prompt(
            task.case_id,
            task.annotator,
            turn_block,
            max_caps=max_caps,
            min_caps=min_caps,
            retry_hint=retry_hint,
        )
        schema_obj = synthetic_seed_schema(max_caps=max_caps, min_caps=min_caps)
        attempt_temperature = temperature if attempt_idx == 0 else min(0.3, temperature + 0.1 * attempt_idx)
        parsed = parse_llm_json(
            client,
            model=model,
            prompt=prompt,
            schema_obj=schema_obj,
            max_tokens=max_tokens,
            temperature=attempt_temperature,
        )
        caps = normalize_seed_caps(parsed, max_caps=max_caps)
        if len(caps) >= min_caps:
            break
    return {
        "case_id": task.case_id,
        "annotator": task.annotator,
        "seed_gold_caps": caps,
        "generator_model": model,
        "min_caps_target": min_caps,
        "attempts_used": attempts_used,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return base.read_jsonl(path)


def extract_transcript(row: Dict[str, Any]) -> str:
    transcript = (
        base.safe_text(row.get("transcript"))
        or base.safe_text(row.get("dialogue"))
        or base.safe_text(row.get("conversation"))
    )
    return transcript


def main() -> None:
    args = parse_args()
    output_dir = base.ensure_dir(args.output_dir)
    output_jsonl = args.output_jsonl or (output_dir / DEFAULT_OUTPUT_JSONL.name)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    base_url = infer_base_url(args.model, args.api_base_url)
    api_key = infer_api_key(args.model, args.api_key)

    if args.model.startswith("gpt-") and not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Set --api-key or export OPENAI_API_KEY.")

    cases = base.load_cases(args.cases_path, args.limit, args.case_ids)
    if not cases:
        raise SystemExit(f"No cases loaded from {args.cases_path}.")

    case_ids = list(cases.keys())
    tasks: List[GenerationTask] = []
    for case_id in case_ids:
        transcript = extract_transcript(cases[case_id])
        if not transcript:
            print(f"[WARN] {case_id}: missing transcript; skipping.", flush=True)
            continue
        tasks.append(GenerationTask(case_id=case_id, transcript=transcript, annotator="A"))
        tasks.append(GenerationTask(case_id=case_id, transcript=transcript, annotator="B"))

    existing_rows: List[Dict[str, Any]] = []
    done_keys: set[Tuple[str, str]] = set()
    if args.skip_existing:
        existing_rows = load_existing_rows(output_jsonl)
        for row in existing_rows:
            key = (base.safe_text(row.get("case_id")), base.safe_text(row.get("annotator")))
            if key[0] and key[1]:
                done_keys.add(key)

    pending = [task for task in tasks if (task.case_id, task.annotator) not in done_keys]
    if not pending:
        print(
            f"[INFO] Nothing to do. Existing rows already cover all requested case/annotator pairs in {output_jsonl}",
            flush=True,
        )
        return

    print(
        f"[INFO] Generating synthetic seed CAPs: cases={len(case_ids)} tasks={len(tasks)} pending={len(pending)} "
        f"model={args.model} base_url={base_url} output={output_jsonl}",
        flush=True,
    )

    generated_rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.task_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                generate_one,
                task,
                model=args.model,
                base_url=base_url,
                api_key=api_key,
                timeout=args.request_timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_transcript_chars=args.max_transcript_chars,
                max_caps_a=args.max_caps_a,
                max_caps_b=args.max_caps_b,
                min_caps_a=args.min_caps_a,
                min_caps_b=args.min_caps_b,
                max_retries_on_thin_output=args.max_retries_on_thin_output,
            ): task
            for task in pending
        }
        completed = 0
        total = len(future_map)
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            completed += 1
            try:
                row = future.result()
                generated_rows.append(row)
                print(
                    f"[INFO] ({completed}/{total}) {task.case_id}@@{task.annotator}: "
                    f"generated {len(row.get('seed_gold_caps', []))} caps",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[ERROR] ({completed}/{total}) {task.case_id}@@{task.annotator}: {exc}",
                    flush=True,
                )

    final_rows = existing_rows + generated_rows if args.skip_existing else generated_rows
    final_rows = sorted(
        final_rows,
        key=lambda row: (base.safe_text(row.get("case_id")), base.safe_text(row.get("annotator"))),
    )
    base.write_jsonl(output_jsonl, final_rows)

    if args.write_case_files:
        case_dir = base.ensure_dir(output_dir / "cases")
        for row in generated_rows:
            case_id = base.safe_text(row.get("case_id"))
            annotator = base.safe_text(row.get("annotator"))
            if not case_id or not annotator:
                continue
            base.write_json(case_dir / f"{case_id}@@{annotator}.json", row)

    print(
        f"[INFO] Wrote {len(generated_rows)} new rows "
        f"(total rows in file: {len(final_rows)}) to {output_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
