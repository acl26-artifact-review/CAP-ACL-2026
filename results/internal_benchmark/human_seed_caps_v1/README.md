# Human Seed CAP v1

This folder contains an initial internal dev benchmark seed for CAP extraction stabilization.

- Cases: `D2N008`, `D2N026`
- Source: clinician-provided A/B annotations shared on 2026-04-11
- Representation: distilled high-salience CAP set for fast iteration
- Intended use: compare CAP extraction variants over time (trend diagnostics), not final benchmark reporting

## Files

- `D2N008.json`
- `D2N026.json`

Each file uses:

- `case_id`
- `seed_gold_caps` (list)
  - `cap_id`
  - `cap_type`
  - `canonical_concept`
  - `proposition_text`
  - `verification_status`
  - `clinical_status`
- `temporality`
- Optional provenance fields for turn-level grounding diagnostics:
  - `provenance_sentence` (e.g., `"4"` or `"6-7"`)
  - `provenance_turn_id` / `provenance_turn_ids`

When provenance fields are present, the benchmark also reports:

- `provenance_turn_overlap_hit_rate_on_matched`
- `provenance_turn_exact_match_rate_on_matched`
- `provenance_turn_jaccard_on_matched`

## Run

```bash
python3 run_cap_internal_benchmark.py \
  --pred-caps-dir outputs/problem_state_tracking_full_207_v23_care_bundle_gemma_100/transcript_caps \
  --gold-caps-dir internal_benchmark/human_seed_caps_v1 \
  --case-ids D2N008 D2N026 \
  --match-threshold 0.45 \
  --write-unmatched \
  --output-dir outputs/cap_internal_benchmark_human_seed_v1
```
