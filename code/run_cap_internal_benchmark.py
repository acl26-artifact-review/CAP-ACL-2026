from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import run_template_rendering_experiments as base


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "cap_internal_benchmark_dev"
CURRENT_YEAR = 2026


@dataclass
class CanonicalCap:
    cap_id: str
    cap_type: str
    concept: str
    proposition_text: str
    verification: str
    clinical_state: str
    temporality_bucket: str
    medication_role: str
    evidence_turn_ids: Tuple[int, ...]


@dataclass
class GoldRow:
    case_id: str
    annotator: str
    caps: List[CanonicalCap]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CAP extraction quality against internal gold/silver CAP labels. "
            "Designed for dev diagnostics and stabilization tracking."
        )
    )
    parser.add_argument("--pred-caps-dir", type=Path, required=True, help="Directory containing predicted CAP JSON files (e.g., transcript_caps/*.json).")
    parser.add_argument("--gold-caps-dir", type=Path, default=None, help="Directory containing gold CAP JSON files keyed by case id.")
    parser.add_argument("--gold-caps-jsonl", type=Path, default=None, help="JSONL file containing gold CAP rows with case_id.")
    parser.add_argument(
        "--gold-merge-mode",
        choices=("union", "intersection", "per_annotator"),
        default="union",
        help=(
            "How to merge multi-annotator gold rows from JSONL. "
            "'union' (default) keeps all unique facts, "
            "'intersection' keeps only facts shared across annotators, "
            "'per_annotator' evaluates separately per annotator."
        ),
    )
    parser.add_argument("--case-ids", nargs="*", default=None, help="Optional explicit case ids.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match-threshold", type=float, default=0.45)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-unmatched", action="store_true", help="Write unmatched CAP diagnostics per case.")
    return parser.parse_args()


def normalize_cap_type(value: Any) -> str:
    raw = base.normalize_text(base.safe_text(value))
    mapping = {
        "chief complaint": "ChiefComplaint",
        "chiefcomplaint": "ChiefComplaint",
        "problemhistory": "ProblemHistory",
        "problem history": "ProblemHistory",
        "examfinding": "ExamFinding",
        "exam finding": "ExamFinding",
        "testresult": "TestResult",
        "test result": "TestResult",
        "medicationstatement": "MedicationStatement",
        "medication statement": "MedicationStatement",
        "medicationrequest": "MedicationRequest",
        "medication request": "MedicationRequest",
        "follow up": "FollowUp",
        "followup": "FollowUp",
        # Human seed CAP labels
        "demographics": "ProblemHistory",
        "past medical history": "ProblemHistory",
        "pastmedicalhistory": "ProblemHistory",
        "social history": "ProblemHistory",
        "socialhistory": "ProblemHistory",
        "lifestyle": "ProblemHistory",
        "symptom": "Symptom",
        "review of systems": "Symptom",
        "reviewofsystems": "Symptom",
        "finding": "ExamFinding",
        "physical exam": "ExamFinding",
        "test result": "TestResult",
        "testresult": "TestResult",
        "assessment": "Diagnosis",
        "diagnosis": "Diagnosis",
        "impression": "Diagnosis",
        "procedure": "Order",
        "plan": "Order",
        "education": "Counseling",
        "counseling": "Counseling",
        "medication": "MedicationStatement",
        "medication plan": "MedicationRequest",
        "medicationplan": "MedicationRequest",
    }
    if raw in mapping:
        return mapping[raw]
    if not raw:
        return "Unknown"
    # Preserve camel-like names when already close to target schema.
    return base.safe_text(value).strip() or "Unknown"


def normalize_verification(item: Dict[str, Any]) -> str:
    verification = base.normalize_text(base.safe_text(item.get("verification_status")))
    status = base.normalize_text(base.safe_text(item.get("status")))
    assertion = base.normalize_text(base.safe_text(item.get("assertion")))
    certainty = base.normalize_text(base.safe_text(item.get("certainty")))
    text = base.normalize_text(
        base.safe_text(item.get("proposition_text")) or base.safe_text(item.get("fact_text"))
    )
    label = verification or assertion or status
    if label in {"confirmed", "affirmed", "present", "positive", "yes"}:
        return "confirmed"
    if label in {"refuted", "negated", "negative", "absent", "denied", "no"}:
        return "refuted"
    if label in {"historical", "history"}:
        return "confirmed"
    # Human annotation files often provide certainty but no explicit verification label.
    # In those cases, infer negation from fact text.
    if any(tok in text for tok in (" denies ", " deny ", " denied ", " no ", " not ", " without ")):
        return "refuted"
    if certainty in {"high", "medium", "low"}:
        return "confirmed"
    return "unconfirmed"


def normalize_clinical_state(item: Dict[str, Any]) -> str:
    state = base.normalize_text(base.safe_text(item.get("clinical_status") or item.get("visit_state")))
    temporality = base.normalize_text(base.safe_text(item.get("temporality")))
    if state in {"planned", "plan"}:
        return "planned"
    if state in {"historical", "history", "past"}:
        return "historical"
    if state in {"resolved"}:
        return "resolved"
    if state in {"active", "stable", "improving", "worsening"}:
        return "active"
    if "future" in temporality:
        return "planned"
    if "past" in temporality:
        return "historical"
    return "active"


def bucket_temporality(item: Dict[str, Any]) -> str:
    t = base.normalize_text(base.safe_text(item.get("temporality")))
    if not t:
        return "unknown"
    if any(token in t for token in ("future", "follow-up", "follow up", "tomorrow", "next", "upcoming", "will")):
        return "future"
    if any(token in t for token in ("current", "today", "now", "ongoing", "present")):
        return "current"
    if any(token in t for token in ("past", "previous", "prior", "ago", "history", "historical", "last", "since", "earlier", "former")):
        return "past"
    year_match = re.search(r"\b(19|20)\d{2}\b", t)
    if year_match:
        year = int(year_match.group(0))
        if year < CURRENT_YEAR:
            return "past"
        if year > CURRENT_YEAR:
            return "future"
        return "current"
    return "unknown"


def normalize_concept(item: Dict[str, Any]) -> str:
    concept = base.safe_text(item.get("canonical_concept"))
    if concept:
        return base.normalize_text(concept)
    text = base.safe_text(item.get("proposition_text")) or base.safe_text(item.get("fact_text"))
    return base.normalize_text(text)


def infer_medication_role(cap_type: str, clinical_state: str) -> str:
    cap_type_norm = base.normalize_text(cap_type)
    if cap_type_norm == base.normalize_text("MedicationStatement"):
        return "state"
    if cap_type_norm in {
        base.normalize_text("MedicationRequest"),
        base.normalize_text("Order"),
        base.normalize_text("FollowUp"),
        base.normalize_text("Counseling"),
    }:
        return "plan"
    if clinical_state == "planned":
        return "plan"
    return "none"


def normalize_turn_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = base.safe_text(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"[Tt]?(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def parse_turn_span(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            out.extend(parse_turn_span(item))
        return out
    text = base.safe_text(value).strip()
    if not text:
        return []

    normalized = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("~", "-")
        .replace(" to ", "-")
        .replace("TO", "-")
    )
    parts = re.split(r"[,;/|]+", normalized)
    out: List[int] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"[Tt]?(\d+)\s*-\s*[Tt]?(\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                start, end = end, start
            if end - start <= 50:
                out.extend(range(start, end + 1))
            else:
                out.extend([start, end])
            continue
        single = normalize_turn_id(token)
        if single is not None:
            out.append(single)
            continue
        out.extend(int(m) for m in re.findall(r"[Tt]?(\d+)", token))
    return out


def extract_evidence_turn_ids(item: Dict[str, Any]) -> Tuple[int, ...]:
    turn_ids: set[int] = set()

    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            turn_id = normalize_turn_id(ev.get("turn_id"))
            if turn_id is not None:
                turn_ids.add(turn_id)

    # Human label path: provenance sentence index from transcript turns.
    for key in ("provenance_sentence", "provenance_turn_id", "provenance_turn_ids", "turn_id", "turn_ids"):
        value = item.get(key)
        if value is None:
            continue
        for turn_id in parse_turn_span(value):
            if turn_id >= 0:
                turn_ids.add(turn_id)

    return tuple(sorted(turn_ids))


def to_canonical_cap(item: Dict[str, Any], idx: int) -> CanonicalCap:
    cap_type = normalize_cap_type(item.get("cap_type") or item.get("category"))
    concept = normalize_concept(item)
    proposition_text = base.safe_text(item.get("proposition_text")) or base.safe_text(item.get("fact_text"))
    verification = normalize_verification(item)
    clinical_state = normalize_clinical_state(item)
    temporality_bucket = bucket_temporality(item)
    medication_role = infer_medication_role(cap_type, clinical_state)
    evidence_turn_ids = extract_evidence_turn_ids(item)
    cap_id = base.safe_text(item.get("cap_id") or item.get("prop_id") or f"CAP{idx}")
    return CanonicalCap(
        cap_id=cap_id,
        cap_type=cap_type,
        concept=concept,
        proposition_text=proposition_text,
        verification=verification,
        clinical_state=clinical_state,
        temporality_bucket=temporality_bucket,
        medication_role=medication_role,
        evidence_turn_ids=evidence_turn_ids,
    )


def extract_cap_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("caps"), list):
        return [x for x in payload["caps"] if isinstance(x, dict)]
    if isinstance(payload.get("facts"), list):
        return [x for x in payload["facts"] if isinstance(x, dict)]
    if isinstance(payload.get("clinical_atomic_facts"), list):
        return [x for x in payload["clinical_atomic_facts"] if isinstance(x, dict)]
    if isinstance(payload.get("atomic_propositions"), list):
        return [x for x in payload["atomic_propositions"] if isinstance(x, dict)]
    if isinstance(payload.get("gold_caps"), list):
        return [x for x in payload["gold_caps"] if isinstance(x, dict)]
    if isinstance(payload.get("seed_gold_caps"), list):
        return [x for x in payload["seed_gold_caps"] if isinstance(x, dict)]
    tf = payload.get("transcript_facts")
    if isinstance(tf, str):
        try:
            parsed = json.loads(tf)
            return extract_cap_list(parsed)
        except Exception:
            return []
    if isinstance(tf, dict):
        return extract_cap_list(tf)
    return []


def load_caps_dir(path: Path) -> Dict[str, List[CanonicalCap]]:
    out: Dict[str, List[CanonicalCap]] = {}
    for file in sorted(path.glob("*.json")):
        case_id = file.stem
        payload = base.read_json(file)
        caps = [to_canonical_cap(item, idx) for idx, item in enumerate(extract_cap_list(payload), start=1)]
        out[case_id] = caps
    return out


def load_caps_jsonl(path: Path) -> Dict[str, List[CanonicalCap]]:
    out: Dict[str, List[CanonicalCap]] = {}
    rows = base.read_jsonl(path)
    for row in rows:
        case_id = base.safe_text(row.get("case_id"))
        if not case_id:
            continue
        caps_raw = extract_cap_list(row)
        out[case_id] = [to_canonical_cap(item, idx) for idx, item in enumerate(caps_raw, start=1)]
    return out


def load_gold_rows_jsonl(path: Path) -> List[GoldRow]:
    rows = base.read_jsonl(path)
    out: List[GoldRow] = []
    for row in rows:
        case_id = base.safe_text(row.get("case_id"))
        if not case_id:
            continue
        annotator = base.safe_text(row.get("annotator") or row.get("rater") or row.get("labeler") or "unknown")
        caps_raw = extract_cap_list(row)
        caps = [to_canonical_cap(item, idx) for idx, item in enumerate(caps_raw, start=1)]
        out.append(GoldRow(case_id=case_id, annotator=annotator, caps=caps))
    return out


def cap_key_for_merge(cap: CanonicalCap) -> Tuple[str, str, str, str]:
    # Merge key keeps clinically relevant core slots while tolerating textual variation.
    return (
        base.normalize_text(cap.cap_type),
        base.normalize_text(cap.concept),
        cap.verification,
        cap.temporality_bucket,
    )


def merge_gold_rows(
    rows: Sequence[GoldRow],
    mode: str,
) -> Dict[str, List[CanonicalCap]]:
    by_case: Dict[str, List[GoldRow]] = {}
    for row in rows:
        by_case.setdefault(row.case_id, []).append(row)

    merged: Dict[str, List[CanonicalCap]] = {}
    for case_id, case_rows in by_case.items():
        if mode == "per_annotator":
            for row in case_rows:
                merged[f"{case_id}@@{row.annotator}"] = row.caps
            continue

        key_counts: Dict[Tuple[str, str, str, str], int] = {}
        key_cap: Dict[Tuple[str, str, str, str], CanonicalCap] = {}
        for row in case_rows:
            seen_keys_in_row = set()
            for cap in row.caps:
                k = cap_key_for_merge(cap)
                if k in seen_keys_in_row:
                    continue
                seen_keys_in_row.add(k)
                key_counts[k] = key_counts.get(k, 0) + 1
                if k not in key_cap:
                    key_cap[k] = cap

        if mode == "union":
            selected_keys = [k for k in key_cap]
        else:
            required = len(case_rows)
            selected_keys = [k for k, c in key_counts.items() if c >= required]

        selected_caps = [key_cap[k] for k in sorted(selected_keys)]
        merged[case_id] = selected_caps
    return merged


def warn_per_annotator_gold_health(gold_map: Dict[str, List[CanonicalCap]]) -> None:
    by_case: Dict[str, Dict[str, List[CanonicalCap]]] = {}
    for key, caps in gold_map.items():
        if "@@" not in key:
            continue
        case_id, annotator = key.split("@@", 1)
        by_case.setdefault(case_id, {})[annotator] = caps

    for case_id in sorted(by_case):
        ann_map = by_case[case_id]
        annotators = sorted(ann_map)
        if len(annotators) < 2:
            print(
                f"[WARN] per_annotator mode: {case_id} has only {len(annotators)} annotator row(s): {annotators}",
                flush=True,
            )
            continue

        signature_by_annotator: Dict[str, set[Tuple[str, str, str, str, str, str]]] = {}
        for annotator, caps in ann_map.items():
            signature: set[Tuple[str, str, str, str, str, str]] = set()
            for cap in caps:
                signature.add(
                    (
                        base.normalize_text(cap.cap_type),
                        base.normalize_text(cap.concept),
                        base.normalize_text(cap.proposition_text),
                        cap.verification,
                        cap.clinical_state,
                        cap.temporality_bucket,
                    )
                )
            signature_by_annotator[annotator] = signature

        unique_signatures = {frozenset(v) for v in signature_by_annotator.values()}
        if len(unique_signatures) == 1:
            print(
                f"[WARN] per_annotator mode: {case_id} annotator CAP sets are identical; check gold JSONL mapping.",
                flush=True,
            )


def cap_similarity(pred: CanonicalCap, gold: CanonicalCap) -> float:
    type_bonus = 1.0 if pred.cap_type == gold.cap_type else 0.0
    concept_sim = base.token_f1(pred.concept, gold.concept)
    prop_sim = base.token_f1(pred.proposition_text, gold.proposition_text)
    return 0.55 * concept_sim + 0.35 * prop_sim + 0.10 * type_bonus


def greedy_match(pred_caps: Sequence[CanonicalCap], gold_caps: Sequence[CanonicalCap], threshold: float) -> List[Tuple[int, int, float]]:
    scored_pairs: List[Tuple[float, int, int]] = []
    for i, pred in enumerate(pred_caps):
        for j, gold in enumerate(gold_caps):
            score = cap_similarity(pred, gold)
            if score >= threshold:
                scored_pairs.append((score, i, j))
    scored_pairs.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for score, i, j in scored_pairs:
        if i in matched_pred or j in matched_gold:
            continue
        matched_pred.add(i)
        matched_gold.add(j)
        matches.append((i, j, score))
    return matches


def prf(matched: int, pred_n: int, gold_n: int) -> Tuple[float, float, float]:
    precision = matched / pred_n if pred_n else 0.0
    recall = matched / gold_n if gold_n else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(statistics.mean(vals), 4)


def evaluate_case(pred_caps: Sequence[CanonicalCap], gold_caps: Sequence[CanonicalCap], threshold: float) -> Dict[str, Any]:
    pred_n = len(pred_caps)
    gold_n = len(gold_caps)
    matches = greedy_match(pred_caps, gold_caps, threshold)
    matched_n = len(matches)
    concept_p, concept_r, concept_f1 = prf(matched_n, pred_n, gold_n)

    verification_correct = 0
    state_correct = 0
    temporality_correct = 0
    full_state_correct = 0
    med_role_total = 0
    med_role_correct = 0
    provenance_total = 0
    provenance_hit = 0
    provenance_exact = 0
    provenance_jaccard_sum = 0.0
    for i, j, _score in matches:
        pred = pred_caps[i]
        gold = gold_caps[j]
        if pred.verification == gold.verification:
            verification_correct += 1
        if pred.clinical_state == gold.clinical_state:
            state_correct += 1
        if pred.temporality_bucket == gold.temporality_bucket:
            temporality_correct += 1
        if (
            pred.verification == gold.verification
            and pred.clinical_state == gold.clinical_state
            and pred.temporality_bucket == gold.temporality_bucket
        ):
            full_state_correct += 1
        if pred.medication_role != "none" or gold.medication_role != "none":
            med_role_total += 1
            if pred.medication_role == gold.medication_role:
                med_role_correct += 1
        if gold.evidence_turn_ids:
            provenance_total += 1
            pred_turns = set(pred.evidence_turn_ids)
            gold_turns = set(gold.evidence_turn_ids)
            overlap = pred_turns & gold_turns
            union = pred_turns | gold_turns
            if overlap:
                provenance_hit += 1
            if pred_turns == gold_turns and pred_turns:
                provenance_exact += 1
            if union:
                provenance_jaccard_sum += len(overlap) / len(union)
            else:
                provenance_jaccard_sum += 0.0

    state_p, state_r, state_f1 = prf(full_state_correct, pred_n, gold_n)
    verification_acc = verification_correct / matched_n if matched_n else 0.0
    clinical_state_acc = state_correct / matched_n if matched_n else 0.0
    temporality_acc = temporality_correct / matched_n if matched_n else 0.0
    medication_role_acc = med_role_correct / med_role_total if med_role_total else None
    provenance_hit_rate = provenance_hit / provenance_total if provenance_total else None
    provenance_exact_rate = provenance_exact / provenance_total if provenance_total else None
    provenance_jaccard = provenance_jaccard_sum / provenance_total if provenance_total else None

    unsupported_pred = pred_n - matched_n
    missing_gold = gold_n - matched_n

    return {
        "pred_cap_count": pred_n,
        "gold_cap_count": gold_n,
        "matched_cap_count": matched_n,
        "unsupported_pred_count": unsupported_pred,
        "missing_gold_count": missing_gold,
        "concept_precision": round(concept_p, 4),
        "concept_recall": round(concept_r, 4),
        "concept_f1": round(concept_f1, 4),
        "state_precision": round(state_p, 4),
        "state_recall": round(state_r, 4),
        "state_f1": round(state_f1, 4),
        "verification_acc_on_matched": round(verification_acc, 4),
        "clinical_state_acc_on_matched": round(clinical_state_acc, 4),
        "temporality_acc_on_matched": round(temporality_acc, 4),
        "medication_state_plan_acc_on_matched": round(medication_role_acc, 4) if medication_role_acc is not None else None,
        "provenance_eval_match_count": provenance_total,
        "provenance_turn_overlap_hit_rate_on_matched": round(provenance_hit_rate, 4) if provenance_hit_rate is not None else None,
        "provenance_turn_exact_match_rate_on_matched": round(provenance_exact_rate, 4) if provenance_exact_rate is not None else None,
        "provenance_turn_jaccard_on_matched": round(provenance_jaccard, 4) if provenance_jaccard is not None else None,
        "_matches": matches,
    }


CASE_FIELDS = [
    "case_id",
    "pred_cap_count",
    "gold_cap_count",
    "matched_cap_count",
    "unsupported_pred_count",
    "missing_gold_count",
    "concept_precision",
    "concept_recall",
    "concept_f1",
    "state_precision",
    "state_recall",
    "state_f1",
    "verification_acc_on_matched",
    "clinical_state_acc_on_matched",
    "temporality_acc_on_matched",
    "medication_state_plan_acc_on_matched",
    "provenance_eval_match_count",
    "provenance_turn_overlap_hit_rate_on_matched",
    "provenance_turn_exact_match_rate_on_matched",
    "provenance_turn_jaccard_on_matched",
]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {"n_cases": len(rows)}
    for metric in CASE_FIELDS:
        if metric == "case_id":
            continue
        values = [row.get(metric) for row in rows]
        aggregate[metric] = safe_mean([float(v) for v in values if v is not None])
    return aggregate


def write_unmatched_case(
    output_dir: Path,
    case_id: str,
    pred_caps: Sequence[CanonicalCap],
    gold_caps: Sequence[CanonicalCap],
    matches: Sequence[Tuple[int, int, float]],
) -> None:
    matched_pred_ids = {i for i, _j, _s in matches}
    matched_gold_ids = {j for _i, j, _s in matches}
    payload = {
        "case_id": case_id,
        "matched_pairs": [
            {
                "pred_idx": i,
                "gold_idx": j,
                "score": round(score, 4),
                "pred_cap": pred_caps[i].__dict__,
                "gold_cap": gold_caps[j].__dict__,
            }
            for i, j, score in matches
        ],
        "unmatched_pred": [pred_caps[i].__dict__ for i in range(len(pred_caps)) if i not in matched_pred_ids],
        "unmatched_gold": [gold_caps[j].__dict__ for j in range(len(gold_caps)) if j not in matched_gold_ids],
    }
    base.write_json(output_dir / f"{case_id}.json", payload)


def main() -> None:
    args = parse_args()
    if not args.gold_caps_dir and not args.gold_caps_jsonl:
        raise SystemExit("Provide at least one of --gold-caps-dir or --gold-caps-jsonl.")

    pred_map = load_caps_dir(args.pred_caps_dir)
    gold_map: Dict[str, List[CanonicalCap]] = {}
    if args.gold_caps_dir:
        gold_map.update(load_caps_dir(args.gold_caps_dir))
    if args.gold_caps_jsonl:
        gold_rows = load_gold_rows_jsonl(args.gold_caps_jsonl)
        if gold_rows:
            merged = merge_gold_rows(gold_rows, args.gold_merge_mode)
            gold_map.update(merged)
        else:
            # Backward-compatible path for legacy JSONL without annotator rows.
            gold_map.update(load_caps_jsonl(args.gold_caps_jsonl))

    if args.gold_merge_mode == "per_annotator":
        warn_per_annotator_gold_health(gold_map)

    if args.gold_merge_mode == "per_annotator":
        case_ids = sorted(
            case_id
            for case_id in gold_map
            if case_id.split("@@", 1)[0] in pred_map
        )
    else:
        case_ids = sorted(set(pred_map) & set(gold_map))

    if args.case_ids:
        requested = {base.safe_text(x) for x in args.case_ids if base.safe_text(x)}
        if args.gold_merge_mode == "per_annotator":
            case_ids = [
                case_id
                for case_id in case_ids
                if case_id in requested or case_id.split("@@", 1)[0] in requested
            ]
        else:
            case_ids = [case_id for case_id in case_ids if case_id in requested]
    if args.limit is not None:
        case_ids = case_ids[: max(0, args.limit)]
    if not case_ids:
        raise SystemExit("No overlapping case ids between predicted CAPs and gold CAPs.")

    output_dir = base.ensure_dir(args.output_dir)
    unmatched_dir = base.ensure_dir(output_dir / "unmatched") if args.write_unmatched else None

    rows: List[Dict[str, Any]] = []
    print(
        f"[INFO] Running CAP internal benchmark cases={len(case_ids)} "
        f"pred_dir={args.pred_caps_dir} "
        f"gold_dir={args.gold_caps_dir} gold_jsonl={args.gold_caps_jsonl} merge_mode={args.gold_merge_mode} "
        f"threshold={args.match_threshold}",
        flush=True,
    )

    for idx, case_id in enumerate(case_ids, start=1):
        if args.gold_merge_mode == "per_annotator":
            base_case_id = case_id.split("@@", 1)[0]
            pred_caps = pred_map.get(base_case_id, [])
            gold_caps = gold_map.get(case_id, [])
        else:
            pred_caps = pred_map.get(case_id, [])
            gold_caps = gold_map.get(case_id, [])
        metrics = evaluate_case(pred_caps, gold_caps, args.match_threshold)
        row = {"case_id": case_id, **{k: v for k, v in metrics.items() if not k.startswith("_")}}
        rows.append(row)
        print(
            f"[INFO] ({idx}/{len(case_ids)}) {case_id}: "
            f"concept_f1={row['concept_f1']:.4f} state_f1={row['state_f1']:.4f} "
            f"prov_hit={row.get('provenance_turn_overlap_hit_rate_on_matched')} "
            f"pred={row['pred_cap_count']} gold={row['gold_cap_count']} matched={row['matched_cap_count']}",
            flush=True,
        )
        if unmatched_dir is not None:
            write_unmatched_case(unmatched_dir, case_id, pred_caps, gold_caps, metrics["_matches"])

    aggregate = build_aggregate(rows)
    base.write_jsonl(output_dir / "case_metrics.jsonl", rows)
    write_csv(output_dir / "case_metrics.csv", rows, CASE_FIELDS)
    base.write_json(output_dir / "aggregate_metrics.json", aggregate)
    write_csv(output_dir / "aggregate_metrics.csv", [aggregate], ["n_cases"] + CASE_FIELDS[1:])
    print(f"[INFO] Wrote CAP internal benchmark results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
