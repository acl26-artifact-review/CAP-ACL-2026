import argparse
import copy
import json
import os
import re
from typing import Dict, List


SUBJECTIVE_HEADER_PATTERNS = [
    re.compile(r"^\s*chief complaint\b", re.IGNORECASE),
    re.compile(r"^\s*history of present illness\b", re.IGNORECASE),
    re.compile(r"^\s*hpi\b", re.IGNORECASE),
    re.compile(r"^\s*history\b", re.IGNORECASE),
    re.compile(r"^\s*medical history\b", re.IGNORECASE),
    re.compile(r"^\s*past medical history\b", re.IGNORECASE),
    re.compile(r"^\s*past surgical history\b", re.IGNORECASE),
    re.compile(r"^\s*past history\b", re.IGNORECASE),
    re.compile(r"^\s*family history\b", re.IGNORECASE),
    re.compile(r"^\s*social history\b", re.IGNORECASE),
    re.compile(r"^\s*medications?\b", re.IGNORECASE),
    re.compile(r"^\s*current medications?\b", re.IGNORECASE),
    re.compile(r"^\s*allerg(?:y|ies)\b", re.IGNORECASE),
    re.compile(r"^\s*review of systems\b", re.IGNORECASE),
]

NON_SUBJECTIVE_HEADER_PATTERNS = [
    re.compile(r"^\s*objective\b", re.IGNORECASE),
    re.compile(r"^\s*physical exam", re.IGNORECASE),
    re.compile(r"^\s*exam\b", re.IGNORECASE),
    re.compile(r"^\s*vitals?(?: reviewed)?\b", re.IGNORECASE),
    re.compile(r"^\s*results?\b", re.IGNORECASE),
    re.compile(r"^\s*labs?\b", re.IGNORECASE),
    re.compile(r"^\s*imaging\b", re.IGNORECASE),
    re.compile(r"^\s*assessment(?: and plan)?\b", re.IGNORECASE),
    re.compile(r"^\s*plan\b", re.IGNORECASE),
    re.compile(r"^\s*instructions?\b", re.IGNORECASE),
]


def _is_subjective_header(line: str) -> bool:
    return any(p.search(line or "") for p in SUBJECTIVE_HEADER_PATTERNS)


def _is_non_subjective_header(line: str) -> bool:
    return any(p.search(line or "") for p in NON_SUBJECTIVE_HEADER_PATTERNS)


def extract_subjective_only(note: str) -> str:
    lines = (note or "").splitlines()
    kept: List[str] = []
    in_subjective = False

    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()
        if _is_subjective_header(stripped):
            in_subjective = True
            kept.append(raw)
            continue
        if _is_non_subjective_header(stripped):
            in_subjective = False
            continue
        if in_subjective:
            kept.append(raw)

    return "\n".join(kept).strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def sentence_in_subjective_view(note: str, sentence: str) -> bool:
    note_view = extract_subjective_only(note)
    norm_sentence = _normalize_text(sentence)
    return bool(norm_sentence) and norm_sentence in _normalize_text(note_view)


def is_s_only_omission_injection(case: Dict, injection: Dict) -> bool:
    if str(injection.get("axis", "")).lower() != "omission":
        return False
    sentence = (
        (injection.get("type_reason_sentence") or {}).get("Sentence")
        or injection.get("original_sentence")
        or injection.get("corrupted_sentence")
        or ""
    )
    return sentence_in_subjective_view(case.get("summary_gt_note", ""), sentence)


def build_s_only_case(case: Dict) -> Dict:
    new_case = copy.deepcopy(case)
    new_case["summary_gt_note"] = extract_subjective_only(case.get("summary_gt_note", ""))
    new_case["summary_draft"] = extract_subjective_only(case.get("summary_draft", ""))
    new_case["summary_draft_original"] = extract_subjective_only(case.get("summary_draft_original", ""))

    filtered_injections = [inj for inj in case.get("stress_injections", []) if is_s_only_omission_injection(case, inj)]
    new_case["stress_injections"] = filtered_injections

    meta = copy.deepcopy(case.get("stress_meta", {}))
    target_counts = copy.deepcopy(meta.get("target_counts", {}))
    target_counts["omission"] = sum(1 for inj in filtered_injections if str(inj.get("axis", "")).lower() == "omission")
    target_counts["omission_major"] = sum(
        1
        for inj in filtered_injections
        if str(inj.get("axis", "")).lower() == "omission" and str(inj.get("severity", "")).lower() == "major"
    )
    meta["target_counts"] = target_counts
    meta["subset_view"] = "s_only_omission"
    new_case["stress_meta"] = meta
    return new_case


def build_major_hall_case(case: Dict) -> Dict:
    new_case = copy.deepcopy(case)
    filtered_injections = [
        inj
        for inj in case.get("stress_injections", [])
        if str(inj.get("axis", "")).lower() == "hallucination" and str(inj.get("severity", "")).lower() == "major"
    ]
    new_case["stress_injections"] = filtered_injections

    meta = copy.deepcopy(case.get("stress_meta", {}))
    target_counts = copy.deepcopy(meta.get("target_counts", {}))
    target_counts["hallucination"] = len(filtered_injections)
    target_counts["hallucination_major"] = len(filtered_injections)
    meta["target_counts"] = target_counts
    meta["subset_view"] = "major_hallucination"
    new_case["stress_meta"] = meta
    return new_case


def build_major_omit_case(case: Dict) -> Dict:
    new_case = copy.deepcopy(case)
    filtered_injections = [
        inj
        for inj in case.get("stress_injections", [])
        if str(inj.get("axis", "")).lower() == "omission" and str(inj.get("severity", "")).lower() == "major"
    ]
    new_case["stress_injections"] = filtered_injections

    meta = copy.deepcopy(case.get("stress_meta", {}))
    target_counts = copy.deepcopy(meta.get("target_counts", {}))
    target_counts["omission"] = len(filtered_injections)
    target_counts["omission_major"] = len(filtered_injections)
    meta["target_counts"] = target_counts
    meta["subset_view"] = "major_omission"
    new_case["stress_meta"] = meta
    return new_case


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stress-set subsets for S-only omission, major omission, and major hallucination evaluation.")
    parser.add_argument("--input_path", required=True, help="Path to the original cases.jsonl")
    parser.add_argument("--output_dir", required=True, help="Directory to write subset JSONL files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cases = read_jsonl(args.input_path)

    s_only_cases = [build_s_only_case(case) for case in cases]
    major_omit_cases_all = [build_major_omit_case(case) for case in cases]
    major_hall_cases_all = [build_major_hall_case(case) for case in cases]

    s_only_cases = [
        case for case in s_only_cases
        if any(str(inj.get("axis", "")).lower() == "omission" for inj in case.get("stress_injections", []))
    ]
    major_omit_cases = [
        case for case in major_omit_cases_all
        if int(((case.get("stress_meta") or {}).get("target_counts") or {}).get("omission_major", 0)) > 0
    ]
    major_hall_cases = [
        case for case in major_hall_cases_all
        if int(((case.get("stress_meta") or {}).get("target_counts") or {}).get("hallucination_major", 0)) > 0
    ]

    s_only_path = os.path.join(args.output_dir, "cases_s_only_omission.jsonl")
    major_omit_path = os.path.join(args.output_dir, "cases_major_omission.jsonl")
    major_hall_path = os.path.join(args.output_dir, "cases_major_hallucination.jsonl")

    write_jsonl(s_only_path, s_only_cases)
    write_jsonl(major_omit_path, major_omit_cases)
    write_jsonl(major_hall_path, major_hall_cases)

    summary = {
        "input_path": args.input_path,
        "s_only_output_path": s_only_path,
        "major_omit_output_path": major_omit_path,
        "major_hall_output_path": major_hall_path,
        "num_cases": len(cases),
        "s_only_omission_injections": sum(
            1 for case in s_only_cases for inj in case.get("stress_injections", [])
        ),
        "major_omission_injections": sum(
            1 for case in major_omit_cases for inj in case.get("stress_injections", [])
        ),
        "major_hallucination_injections": sum(
            1 for case in major_hall_cases for inj in case.get("stress_injections", [])
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
