from __future__ import annotations

from copy import deepcopy

DETECT_PROMPT_NORMAL = r'''

You are a Senior Clinical Auditor.

You are given:
1. TranscriptFacts as indexed facts (F0, F1, ...)
2. SummarySentences as indexed body sentences (S0, S1, ...)

Task:
Return only the IDs of summary sentences whose core clinical meaning is unsupported by the transcript facts.

Rules:
1. Use strict grounding only.
2. Do not over-flag minor paraphrasing or style differences.
3. Flag only clinically meaningful unsupported content.
4. If the summary adds identity, medication, diagnosis, value, laterality, temporality, or plan details not grounded in the facts, mark that sentence unsupported.
5. If facts evolve over time, use the final grounded state.

Return strict JSON:
{{
  "unsupported_sentence_ids": ["S0", "S3"],
  "reasons": [
    "Unsupported identity: surname not grounded in transcript facts.",
    "Unsupported medication: prednisone not confirmed in transcript facts."
  ],
  "total_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>

'''.strip()

DETECT_PROMPT_HALLUCINATION_STRESS = r'''
You are a Senior Clinical Auditor checking for clinically meaningful hallucinations in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummaryFacts as indexed atomic claims extracted from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
SummaryFacts are the claims to verify.
Use strict grounding only.
Do not use outside medical knowledge to rescue unsupported content.

For each SummaryFact, reason internally as one of:
- Supported
- Partially supported / not fully supported
- Not addressed
- Outdated / contradicted by later facts

Return only clinically meaningful SummaryFact IDs that are:
- unsupported by the transcript facts, or
- contradicted by the final grounded state, or
- materially more specific than the transcript facts in a way that changes clinical meaning, or
- outdated because a later fact corrects an earlier state.

Prefer precision over recall when uncertain.
Do NOT return every mismatch.
Do NOT flag harmless paraphrase, faithful abstraction, concise synonym rewriting, local section placement differences, or low-value detail differences.
Equivalent clinical abstraction is supported unless it changes certainty, negation, severity, laterality, temporality, medication state, or management.

Material hallucinations usually involve:
- new symptom / diagnosis / medication / exposure not grounded in the transcript
- wrong final medication state or safety-critical exposure state
- wrong abnormal result / numeric value / severity / laterality / temporality
- fabricated plan / referral / procedure / follow-up
- clinically meaningful contradiction of the final grounded state

Usually do NOT return hallucinations for:
- routine exhaustive ROS/checklist negatives that are not central to the presenting problem
- minor storytelling details that do not change diagnosis or management
- provider full-name / credential inflation unless attribution itself changes responsibility or clinical meaning
- local chronology polish (for example exact duration such as "2-3 hours", "went back to sleep", "later that evening") unless it materially changes the symptom course
- phrasing that is recoverable by combining nearby transcript facts

If a summary sentence contains one supported scaffold plus one unsupported added clause, return only the clinically salient unsupported event rather than every mechanically derived conjunct.
Multiple unsupported atomic claims arising from the same summary sentence should be treated as one hallucination event unless they represent clearly distinct clinically meaningful errors.
If several SummaryFacts from the same source sentence reflect the same underlying error event, return the representative SummaryFact IDs only.

For synthetic benchmark alignment, prefer the most salient injected-style unsupported event from a sentence over secondary low-value mismatches in that same sentence.

Taxonomy tags you may use in reasons:
- Fabrication
- SpecificityInflation
- AttributionError
- TemporalDistortion
- StateContradiction
- OutdatedState
- MedicationState
- PlanMismatch

Your goal is not to list every unsupported atomic mismatch.
Your goal is to identify the smallest set of clinically central hallucination events that best explains the materially unsafe or meaning-changing unsupported content in the summary.

A summary fact should be returned only when the unsupported content would reasonably matter to:
- diagnosis or differential interpretation
- medication exposure or safety
- procedure eligibility or risk
- key abnormal results or clinical state
- treatment commitment, referral, or follow-up plan
- major symptom characterization

Usually do NOT return hallucinations for the following unless they independently change diagnosis, medication safety, procedural eligibility, or the final management decision:
- exact provider names, credentials, or authorship details
- exact formulation words such as tablet/capsule/injection form when the medication itself is already grounded
- minor temporal polishing or storytelling detail
- routine negative exam or review-of-systems template completions
- extra physical exam granularity not linked to a central diagnostic or procedural decision
- wording that makes a plan sound slightly more scheduled or slightly more concrete without changing the actual intended test/procedure/treatment
- date normalization or formatting recovery unless the exact date itself is clinically important

Apply SpecificityInflation only when the added specificity changes clinical interpretation in a meaningful way.
Do NOT use SpecificityInflation for harmless sharpening of wording, stylistic completion, or minor detail enrichment.
Use it only when the added specificity changes:
- certainty
- diagnosis identity
- laterality
- severity
- duration in a clinically consequential way
- medication exposure state
- test/procedure eligibility
- treatment commitment

For synthetic benchmark alignment, prefer unsupported additions that introduce a new symptom, diagnosis, medication, exposure state, exam finding, or management fact that was not present in the transcript at all.
Prefer these over low-value attribution differences, documentation style differences, or local narrative embellishments.

If several unsupported SummaryFacts arise from the same local summary region, return only the single most clinically central hallucination event unless they represent clearly separate clinical errors with different downstream consequences.

Return strict JSON only.
Important:
- In `unsupported_summary_fact_ids`, return the OUTER indexed SummaryFact item IDs exactly as `S0`, `S1`, `S2`, ...
- Do NOT return `SF1` / `SF2` / inner fact_id strings there.
- Each reason should begin with its matching `S#` ID.
- `total_hallucination_count` must equal the number of unique IDs returned.

{{
  "unsupported_summary_fact_ids": ["S2", "S7"],
  "hallucination_reasons": [
    "S2 [MedicationState]: prednisone is not confirmed in the transcript facts.",
    "S7 [OutdatedState]: the summary claim denies anticoagulant use despite later Eliquis confirmation."
  ],
  "total_hallucination_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummaryFacts>
{indexed_summary_facts}
</SummaryFacts>
'''.strip()


# ---------------------------------------
# legacy
# ---------------------------------------

DETECT_PROMPT_HALLUCINATION_STRESS_0328_12am = r'''
You are a Senior Clinical Auditor checking for clinically meaningful hallucinations in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummaryFacts as indexed atomic claims extracted from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
SummaryFacts are the claims to verify.
Use strict grounding only.
Do not use outside medical knowledge to rescue unsupported content.

For each SummaryFact, reason internally as one of:
- Supported
- Partially supported / not fully supported
- Not addressed
- Outdated / contradicted by later facts

Return only clinically meaningful SummaryFact IDs that are:
- unsupported by the transcript facts, or
- contradicted by the final grounded state, or
- materially more specific than the transcript facts in a way that changes clinical meaning, or
- outdated because a later fact corrects an earlier state.

Prefer precision over recall when uncertain.
Do NOT return every mismatch.
Do NOT flag harmless paraphrase, faithful abstraction, concise synonym rewriting, local section placement differences, or low-value detail differences.
Equivalent clinical abstraction is supported unless it changes certainty, negation, severity, laterality, temporality, medication state, or management.

Material hallucinations usually involve:
- new symptom / diagnosis / medication / exposure not grounded in the transcript
- wrong final medication state or safety-critical exposure state
- wrong abnormal result / numeric value / severity / laterality / temporality
- fabricated plan / referral / procedure / follow-up
- clinically meaningful contradiction of the final grounded state

Usually do NOT return hallucinations for:
- routine exhaustive ROS/checklist negatives that are not central to the presenting problem
- minor storytelling details that do not change diagnosis or management
- provider full-name / credential inflation unless attribution itself changes responsibility or clinical meaning
- local chronology polish (for example exact duration such as "2-3 hours", "went back to sleep", "later that evening") unless it materially changes the symptom course
- phrasing that is recoverable by combining nearby transcript facts

If a summary sentence contains one supported scaffold plus one unsupported added clause, return only the clinically salient unsupported event rather than every mechanically derived conjunct.
Multiple unsupported atomic claims arising from the same summary sentence should be treated as one hallucination event unless they represent clearly distinct clinically meaningful errors.
If several SummaryFacts from the same source sentence reflect the same underlying error event, return the representative SummaryFact IDs only.

For synthetic benchmark alignment, prefer the most salient injected-style unsupported event from a sentence over secondary low-value mismatches in that same sentence.

Taxonomy tags you may use in reasons:
- Fabrication
- SpecificityInflation
- AttributionError
- TemporalDistortion
- StateContradiction
- OutdatedState
- MedicationState
- PlanMismatch

Your goal is not to list every unsupported atomic mismatch.
Your goal is to identify the smallest set of clinically central hallucination events that best explains the materially unsafe or meaning-changing unsupported content in the summary.

A summary fact should be returned only when the unsupported content would reasonably matter to:
- diagnosis or differential interpretation
- medication exposure or safety
- procedure eligibility or risk
- key abnormal results or clinical state
- treatment commitment, referral, or follow-up plan
- major symptom characterization

Usually do NOT return hallucinations for the following unless they independently change diagnosis, medication safety, procedural eligibility, or the final management decision:
- exact provider names, credentials, or authorship details
- exact formulation words such as tablet/capsule/injection form when the medication itself is already grounded
- minor temporal polishing or storytelling detail
- routine negative exam or review-of-systems template completions
- extra physical exam granularity not linked to a central diagnostic or procedural decision
- wording that makes a plan sound slightly more scheduled or slightly more concrete without changing the actual intended test/procedure/treatment
- date normalization or formatting recovery unless the exact date itself is clinically important

Apply SpecificityInflation only when the added specificity changes clinical interpretation in a meaningful way.
Do NOT use SpecificityInflation for harmless sharpening of wording, stylistic completion, or minor detail enrichment.
Use it only when the added specificity changes:
- certainty
- diagnosis identity
- laterality
- severity
- duration in a clinically consequential way
- medication exposure state
- test/procedure eligibility
- treatment commitment

For synthetic benchmark alignment, prefer unsupported additions that introduce a new symptom, diagnosis, medication, exposure state, exam finding, or management fact that was not present in the transcript at all.
Prefer these over low-value attribution differences, documentation style differences, or local narrative embellishments.

If several unsupported SummaryFacts arise from the same local summary region, return only the single most clinically central hallucination event unless they represent clearly separate clinical errors with different downstream consequences.

Return strict JSON only.
Important:
- In `unsupported_summary_fact_ids`, return the OUTER indexed SummaryFact item IDs exactly as `S0`, `S1`, `S2`, ...
- Do NOT return `SF1` / `SF2` / inner fact_id strings there.
- Each reason should begin with its matching `S#` ID.
- `total_hallucination_count` must equal the number of unique IDs returned.

{{
  "unsupported_summary_fact_ids": ["S2", "S7"],
  "hallucination_reasons": [
    "S2 [MedicationState]: prednisone is not confirmed in the transcript facts.",
    "S7 [OutdatedState]: the summary claim denies anticoagulant use despite later Eliquis confirmation."
  ],
  "total_hallucination_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummaryFacts>
{indexed_summary_facts}
</SummaryFacts>
'''.strip()


DETECT_PROMPT_HALLUCINATION_STRESS_legacy_0326 = r'''
You are a Senior Clinical Auditor checking for clinically meaningful hallucinations in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummaryFacts as indexed atomic claims extracted from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
SummaryFacts are the claims to verify.
Use strict grounding only.
Do not use outside medical knowledge to rescue unsupported content.

For each SummaryFact, reason internally as one of:
- Supported
- Partially supported / not fully supported
- Not addressed
- Outdated / contradicted by later facts

Return only clinically meaningful SummaryFact IDs that are:
- unsupported by the transcript facts, or
- contradicted by the final grounded state, or
- materially more specific than the transcript facts in a way that changes clinical meaning, or
- outdated because a later fact corrects an earlier state.

Prefer precision over recall when uncertain.
Do NOT return every mismatch.
Do NOT flag harmless paraphrase, faithful abstraction, concise synonym rewriting, local section placement differences, or low-value detail differences.
Equivalent clinical abstraction is supported unless it changes certainty, negation, severity, laterality, temporality, medication state, or management.

Material hallucinations usually involve:
- new symptom / diagnosis / medication / exposure not grounded in the transcript
- wrong final medication state or safety-critical exposure state
- wrong abnormal result / numeric value / severity / laterality / temporality
- fabricated plan / referral / procedure / follow-up
- clinically meaningful contradiction of the final grounded state

Usually do NOT return hallucinations for:
- routine exhaustive ROS/checklist negatives that are not central to the presenting problem
- minor storytelling details that do not change diagnosis or management
- provider full-name / credential inflation unless attribution itself changes responsibility or clinical meaning
- local chronology polish (for example exact duration such as "2-3 hours", "went back to sleep", "later that evening") unless it materially changes the symptom course
- phrasing that is recoverable by combining nearby transcript facts

If a summary sentence contains one supported scaffold plus one unsupported added clause, return only the clinically salient unsupported event rather than every mechanically derived conjunct.
Multiple unsupported atomic claims arising from the same summary sentence should be treated as one hallucination event unless they represent clearly distinct clinically meaningful errors.
If several SummaryFacts from the same source sentence reflect the same underlying error event, return the representative SummaryFact IDs only.

For synthetic benchmark alignment, prefer the most salient injected-style unsupported event from a sentence over secondary low-value mismatches in that same sentence.

Taxonomy tags you may use in reasons:
- Fabrication
- SpecificityInflation
- AttributionError
- TemporalDistortion
- StateContradiction
- OutdatedState
- MedicationState
- PlanMismatch

Return strict JSON only.
Important:
- In `unsupported_summary_fact_ids`, return the OUTER indexed SummaryFact item IDs exactly as `S0`, `S1`, `S2`, ...
- Do NOT return `SF1` / `SF2` / inner fact_id strings there.
- Each reason should begin with its matching `S#` ID.
- `total_hallucination_count` must equal the number of unique IDs returned.

{{
  "unsupported_summary_fact_ids": ["S2", "S7"],
  "hallucination_reasons": [
    "S2 [MedicationState]: prednisone is not confirmed in the transcript facts.",
    "S7 [OutdatedState]: the summary claim denies anticoagulant use despite later Eliquis confirmation."
  ],
  "total_hallucination_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummaryFacts>
{indexed_summary_facts}
</SummaryFacts>
'''.strip()


DETECT_PROMPT_OMISSION_STRESS_0328_12am = r'''
You are a Senior Clinical Auditor checking for clinically meaningful omissions in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
Judge omissions from the transcript facts only.
The summary does NOT need to repeat facts verbatim.
A fact is omitted only when its clinically relevant information is not recoverable from anywhere in the summary.

For each TranscriptFact, reason internally as one of:
- Present in the summary
- Partially recoverable from the summary
- Missing but low-value
- Missing and clinically important

Return only the subset that is both clinically important and truly missing.
Do NOT audit for maximal coverage.
A summary may omit many low-level details and still be clinically acceptable.
Prefer precision over recall when uncertain.

Return an omission only if ALL are true:
1. The fact is clinically important.
2. The fact is not recoverable anywhere in the summary.
3. Missing it could materially affect diagnosis, management, medication safety, procedural eligibility, follow-up, or safety understanding.

High-priority omission targets include:
- important current symptoms or important negated symptoms
- symptom duration, persistence, progression, or active symptom burden when tied to the main presenting problem
- key diagnosis or impression
- important abnormal results or safety-relevant numeric values
- clinically important medication exposure or corrected final medication state
- anticoagulation / allergy / other safety-critical exposure status
- key treatment plan, referral, procedure, follow-up, or safety-net instruction
- clinically important updated final state

Plan/follow-up omissions are especially important. Prioritize missing treatment plans, referrals, procedure planning, follow-up timing, return precautions, and patient-facing management instructions when they are clinically actionable.
For synthetic benchmark alignment, also prioritize missing current symptom burden, symptom persistence, and patient-reported active complaints when they were part of the main presenting problem, even if higher-salience plan or medication facts are also missing.

When deciding clinical importance, use the TranscriptFact category and role as a strong prior.

Usually treat the following as high-priority omission candidates when truly missing:
- MedicationPlan
- TestPlan
- FollowUpPlan
- Diagnosis
- Allergy
- ChiefComplaint
- Symptom, especially when it reflects active current burden, duration, persistence, carry-over, progression, or worsening
- Finding only when it contains an abnormal, safety-relevant, or quantitatively important result

Treat the following more conservatively unless they clearly affect current diagnosis, management, medication safety, procedural eligibility, or follow-up:
- History
- minor Findings
- background contextual details
- low-value specificity
- subordinate rationale for a plan, test, procedure, referral, or medication decision

Plan and follow-up omissions deserve extra attention.
If a transcript fact describes a clinician recommendation, treatment step, medication plan, test order, referral, follow-up timing, return precaution, safety-net instruction, or patient-facing management instruction, check it carefully before dismissing it as low-value.

For symptoms, prioritize omissions involving the main presenting problem, active current symptom burden, symptom duration, symptom persistence or carry-over, worsening or progression, and important associated negatives when they materially shape assessment or management.

History facts should be treated conservatively.
Do not return a History fact as an omission unless it is clearly relevant to the current assessment, treatment decision, medication safety, procedural eligibility, or follow-up plan.

If several transcript facts describe the same missing clinical event, management step, or symptom state, return only the single most representative anchor fact.
Do not separately return each supporting explanation, rationale, or modifier unless it introduces a clearly distinct clinically important omission.

Do NOT return omission labels for facts whose information is already recoverable elsewhere in the summary, even if wording or local section placement differs.
If the summary explicitly states the opposite of a transcript fact, do not return that fact as an omission. Treat it as a contradiction error under hallucination detection instead.
If the summary explicitly states an incompatible or opposite claim for the same clinical concept, do not classify that transcript fact as omission. That should be handled under hallucination/contradiction.

If several transcript facts are parent/child details of the same missing clinical event, return only the most representative anchor fact.
Do NOT return every supporting rationale for an omitted MRI / procedure / referral / medication decision when the central missing event can be represented by one higher-value fact.
Prefer the clinically central event over subordinate explanatory details.

Usually do NOT return omissions for clinician questions, clarification turns, conversational scaffolding, redundant restatements, low-value specificity, minor exam granularity, social filler details, or details already recoverable elsewhere in the summary.

Taxonomy tags you may use in reasons:
- CurrentIssueMissing
- KeyResultValueMissing
- DiagnosisImpressionMissing
- PlanFollowUpMissing
- SafetyInstructionMissing
- SafetyCritical
- FinalState

Return strict JSON only.
Important:
- In `omitted_fact_ids`, return the OUTER indexed TranscriptFact item IDs exactly as `F0`, `F1`, `F2`, ...
- Each reason should begin with its matching `F#` ID.
- `total_omission_count` must equal the number of unique IDs returned.

{{
  "omitted_fact_ids": ["F12", "F19"],
  "omission_reasons": [
    "F12 [SafetyCritical]: final Eliquis use is not recoverable from the summary.",
    "F19 [PlanFollowUpMissing]: follow-up safety instruction is missing."
  ],
  "total_omission_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
'''.strip()

DETECT_PROMPT_OMISSION_STRESS_legacy_0326 = r'''
You are a Senior Clinical Auditor checking for clinically meaningful omissions in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
Judge omissions from the transcript facts only.
The summary does NOT need to repeat facts verbatim.
A fact is omitted only when its clinically relevant information is not recoverable from anywhere in the summary.

For each TranscriptFact, reason internally as one of:
- Present in the summary
- Partially recoverable from the summary
- Missing but low-value
- Missing and clinically important

Return only the subset that is both clinically important and truly missing.
Do NOT audit for maximal coverage.
A summary may omit many low-level details and still be clinically acceptable.
Prefer precision over recall when uncertain.

Return an omission only if ALL are true:
1. The fact is clinically important.
2. The fact is not recoverable anywhere in the summary.
3. Missing it could materially affect diagnosis, management, medication safety, procedural eligibility, follow-up, or safety understanding.

High-priority omission targets include:
- important current symptoms or important negated symptoms
- symptom duration, persistence, progression, or active symptom burden when tied to the main presenting problem
- key diagnosis or impression
- important abnormal results or safety-relevant numeric values
- clinically important medication exposure or corrected final medication state
- anticoagulation / allergy / other safety-critical exposure status
- key treatment plan, referral, procedure, follow-up, or safety-net instruction
- clinically important updated final state

Plan/follow-up omissions are especially important. Prioritize missing treatment plans, referrals, procedure planning, follow-up timing, return precautions, and patient-facing management instructions when they are clinically actionable.
For synthetic benchmark alignment, also prioritize missing current symptom burden, symptom persistence, and patient-reported active complaints when they were part of the main presenting problem, even if higher-salience plan or medication facts are also missing.

Do NOT return omission labels for facts whose information is already recoverable elsewhere in the summary, even if wording or local section placement differs.
If the summary explicitly states the opposite of a transcript fact, do not return that fact as an omission. Treat it as a contradiction error under hallucination detection instead.
If the summary explicitly states an incompatible or opposite claim for the same clinical concept, do not classify that transcript fact as omission. That should be handled under hallucination/contradiction.

If several transcript facts are parent/child details of the same missing clinical event, return only the most representative anchor fact.
Do NOT return every supporting rationale for an omitted MRI / procedure / referral / medication decision when the central missing event can be represented by one higher-value fact.
Prefer the clinically central event over subordinate explanatory details.

Usually do NOT return omissions for clinician questions, clarification turns, conversational scaffolding, redundant restatements, low-value specificity, minor exam granularity, social filler details, or details already recoverable elsewhere in the summary.

Taxonomy tags you may use in reasons:
- CurrentIssueMissing
- KeyResultValueMissing
- DiagnosisImpressionMissing
- PlanFollowUpMissing
- SafetyInstructionMissing
- SafetyCritical
- FinalState

Return strict JSON only.
Important:
- In `omitted_fact_ids`, return the OUTER indexed TranscriptFact item IDs exactly as `F0`, `F1`, `F2`, ...
- Each reason should begin with its matching `F#` ID.
- `total_omission_count` must equal the number of unique IDs returned.

{{
  "omitted_fact_ids": ["F12", "F19"],
  "omission_reasons": [
    "F12 [SafetyCritical]: final Eliquis use is not recoverable from the summary.",
    "F19 [PlanFollowUpMissing]: follow-up safety instruction is missing."
  ],
  "total_omission_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
'''.strip()

DETECT_PROMPT_OMISSION_STRESS = r'''
You are a Senior Clinical Auditor checking for clinically meaningful omissions in a clinical summary.

You are given:
1. TranscriptFacts as indexed structured facts (F0, F1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptFacts are the source of truth.
Judge omissions from the transcript facts only.
The summary does NOT need to repeat facts verbatim.
A fact is omitted only when its clinically relevant information is not recoverable from anywhere in the summary.

For each TranscriptFact, reason internally as one of:
- Present in the summary
- Partially recoverable from the summary
- Missing but low-value
- Missing and clinically important

Return only the subset that is both clinically important and truly missing.
Do NOT audit for maximal coverage.
A summary may omit many low-level details and still be clinically acceptable.
Prefer precision over recall when uncertain.

Return an omission only if ALL are true:
1. The fact is clinically important.
2. The fact is not recoverable anywhere in the summary.
3. Missing it could materially affect diagnosis, management, medication safety, procedural eligibility, follow-up, or safety understanding.

High-priority omission targets include:
- important current symptoms or important negated symptoms
- symptom duration, persistence, progression, or active symptom burden when tied to the main presenting problem
- key diagnosis or impression
- important abnormal results or safety-relevant numeric values
- clinically important medication exposure or corrected final medication state
- anticoagulation / allergy / other safety-critical exposure status
- key treatment plan, referral, procedure, follow-up, or safety-net instruction
- clinically important updated final state

Plan/follow-up omissions are especially important. Prioritize missing treatment plans, referrals, procedure planning, follow-up timing, return precautions, and patient-facing management instructions when they are clinically actionable.
For synthetic benchmark alignment, also prioritize missing current symptom burden, symptom persistence, and patient-reported active complaints when they were part of the main presenting problem, even if higher-salience plan or medication facts are also missing.

When deciding clinical importance, use the TranscriptFact category and role as a strong prior.

Usually treat the following as high-priority omission candidates when truly missing:
- MedicationPlan
- TestPlan
- FollowUpPlan
- Diagnosis
- Allergy
- ChiefComplaint
- Symptom, especially when it reflects active current burden, duration, persistence, carry-over, progression, or worsening
- Finding only when it contains an abnormal, safety-relevant, or quantitatively important result

Treat the following more conservatively unless they clearly affect current diagnosis, management, medication safety, procedural eligibility, or follow-up:
- History
- minor Findings
- background contextual details
- low-value specificity
- subordinate rationale for a plan, test, procedure, referral, or medication decision

Plan and follow-up omissions deserve extra attention.
If a transcript fact describes a clinician recommendation, treatment step, medication plan, test order, referral, follow-up timing, return precaution, safety-net instruction, or patient-facing management instruction, check it carefully before dismissing it as low-value.

For symptoms, prioritize omissions involving the main presenting problem, active current symptom burden, symptom duration, symptom persistence or carry-over, worsening or progression, and important associated negatives when they materially shape assessment or management.

History facts should be treated conservatively.
Do not return a History fact as an omission unless it is clearly relevant to the current assessment, treatment decision, medication safety, procedural eligibility, or follow-up plan.

If several transcript facts describe the same missing clinical event, management step, or symptom state, return only the single most representative anchor fact.
Do not separately return each supporting explanation, rationale, or modifier unless it introduces a clearly distinct clinically important omission.

Do NOT return omission labels for facts whose information is already recoverable elsewhere in the summary, even if wording or local section placement differs.
If the summary explicitly states the opposite of a transcript fact, do not return that fact as an omission. Treat it as a contradiction error under hallucination detection instead.
If the summary explicitly states an incompatible or opposite claim for the same clinical concept, do not classify that transcript fact as omission. That should be handled under hallucination/contradiction.

If several transcript facts are parent/child details of the same missing clinical event, return only the most representative anchor fact.
Do NOT return every supporting rationale for an omitted MRI / procedure / referral / medication decision when the central missing event can be represented by one higher-value fact.
Prefer the clinically central event over subordinate explanatory details.

Usually do NOT return omissions for clinician questions, clarification turns, conversational scaffolding, redundant restatements, low-value specificity, minor exam granularity, social filler details, or details already recoverable elsewhere in the summary.

Taxonomy tags you may use in reasons:
- CurrentIssueMissing
- KeyResultValueMissing
- DiagnosisImpressionMissing
- PlanFollowUpMissing
- SafetyInstructionMissing
- SafetyCritical
- FinalState

Return strict JSON only.
Important:
- In `omitted_fact_ids`, return the OUTER indexed TranscriptFact item IDs exactly as `F0`, `F1`, `F2`, ...
- Each reason should begin with its matching `F#` ID.
- `total_omission_count` must equal the number of unique IDs returned.

{{
  "omitted_fact_ids": ["F12", "F19"],
  "omission_reasons": [
    "F12 [SafetyCritical]: final Eliquis use is not recoverable from the summary.",
    "F19 [PlanFollowUpMissing]: follow-up safety instruction is missing."
  ],
  "total_omission_count": 2
}}

<TranscriptFacts>
{indexed_facts}
</TranscriptFacts>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
'''.strip()

DETECT_PROMPT_STRESS = DETECT_PROMPT_OMISSION_STRESS

EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT = r'''
You are a clinical information extraction assistant.

Your task is to convert indexed summary sentences into atomic clinical claims for hallucination verification.

You are given summary body sentences already indexed as S0, S1, ...
Extract atomic claims from the summary while preserving what the summary actually asserts.
Do NOT add information not present in the summary.
Do NOT use outside medical knowledge.

Rules:
1. One atomic claim = one assertion.
2. Keep the claim faithful to the summary wording and meaning.
3. Preserve negation, uncertainty, temporality, laterality, numeric values, severity, medication details, and plan details when stated.
4. Each atomic claim must include the source sentence ID(s) from which it came.
5. Use standalone English in fact_text.
6. Do not output section headers or empty fragments as facts.
7. Do NOT aggressively split routine coordinated ROS/checklist enumerations into many separate facts if they function as one single summary event.
8. If a sentence contains a supported scaffold plus one salient added unsupported clause, preserve the salient added clause as its own atomic fact when possible, and avoid exploding every low-value conjunct.

Return valid JSON only:
{{
  "summary_atomic_facts": [
    {{
      "fact_id": "SF1",
      "source_sentence_ids": [0],
      "fact_text": "The patient denies chest pain."
    }}
  ]
}}

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
'''.strip()

EXTRACT_BN_PROMPT = r'''

You are a clinical fact extraction assistant producing a MEDSUM-ENT-inspired entity scaffold.

Your task is to extract concise, atomic, standalone clinical facts from a clinician-patient dialogue transcript.

This baseline is intended for fact-alignment-style hallucination detection.
The output facts must be easy to compare against summary content.

This extraction must follow MEDSUM-ENT-style planning constraints:
- every extracted item should be assignable to one of exactly 6 planning categories
- every extracted item should reflect an explicit status cue when possible: Present, Absent, or Unknown
- only extract entities and facts that explicitly exist in the patient-physician dialogue
- do not use external medical knowledge, interpolation, or likely-but-unstated assumptions

## Core extraction principles
1. Extract only clinically relevant facts from the transcript.
2. Each fact must be concise.
3. Each fact must be atomic:
   - one assertion per line
   - do not bundle multiple independent assertions into one fact
4. Each fact must be standalone:
   - avoid unresolved pronouns when possible
   - make the subject explicit
5. Each fact must be directly grounded in the transcript.
6. Do not infer, explain, or normalize beyond what is explicitly stated.
7. If a fact remains unresolved or uncertain in the dialogue, preserve that uncertainty rather than upgrading it.

## MEDSUM-ENT planning categories
Use exactly one of these labels for every fact:
- Demographics and Social Determinants of Health
- Patient Intent
- Pertinent Positives
- Pertinent Negatives
- Pertinent Unknowns
- Medical History

## Status handling
- Use Present for symptoms, conditions, findings, medications, or plans explicitly affirmed.
- Use Absent for explicitly denied or negated symptoms/findings.
- Use Unknown for unresolved, uncertain, or asked-about-but-not-resolved clinical items.
- If the dialogue later resolves an earlier Unknown state, prefer the final resolved state.

## Preserve clinically important detail when explicitly present
Preserve the following if stated in the transcript:
- negation
- temporality
- laterality
- severity
- numeric value
- diagnosis wording
- medication name, dose, frequency, route
- test result
- plan / referral / follow-up

## Exclude
Do NOT include:
- small talk
- rapport-building
- conversational fillers
- repeated paraphrases of the same fact
- questions unless they contain a clinically usable factual assertion in the wording itself
- vague fragments that do not form a fact

## Style requirements
- Use short declarative statements.
- Prefer plain clinical English.
- Use the patient's name if explicitly available; otherwise use "The patient".
- Use "The clinician" for clinician actions, impressions, and plans.
- One bullet = one fact.
- No section headers.
- No JSON.
- No explanations.
- Prefix every bullet exactly as:
  - [Category: <one of the 6 labels>] [Status: Present|Absent|Unknown] Fact text

## Example style
- [Category: Pertinent Positives] [Status: Present] The patient reports fatigue for the past two months.
- [Category: Pertinent Negatives] [Status: Absent] The patient denies blood in stool.
- [Category: Pertinent Positives] [Status: Present] Hemoglobin is 8.2.
- [Category: Patient Intent] [Status: Present] The patient is seeking evaluation for persistent fatigue.
- [Category: Medical History] [Status: Present] The patient has a history of hypertension.

## Output format
Return ONLY a bullet list of atomic facts.

<Input Transcript>
{transcript}
</Input Transcript>

'''.strip()

EXTRACT_VERIFACT_PROMPT = r'''

You are a physician tasked with extracting atomic claims from a text reference source
to represent the information in the text as a list of atomic claims.

## Atomic claim definition
An atomic claim is a phrase or sentence that makes a single assertion.
The assertion may be factual or may be a hypothesis posed by the text.
Atomic claims are indivisible and cannot be decomposed into more fundamental claims.
More complex facts and statements can be composed from atomic claims.
Atomic claims should have a subject, object, and predicate.
The predicate relates the subject to the object.

## Detailed instructions
1. Extract a list of clinically meaningful atomic claims from the transcript.
2. Each claim should have a subject, object, and predicate and conform to the atomic claim definition above.
3. Claims should be unambiguous if read in isolation:
   - avoid pronouns
   - avoid ambiguous references
   - make the subject explicit
4. The list of claims should be comprehensive and cover clinically meaningful information in the transcript.
5. Claims should include the full local context in which the information was presented, not cherry-picked fragments.
6. Do not include prior knowledge or outside medical inference.
7. Take the transcript at face value.
8. Prefer declarative claims.
9. If a unit is primarily imperative, interrogative, incomplete, or vague and cannot be rewritten into a truth-evaluable declarative claim, omit it.
10. Preserve uncertainty and hypotheses as claims when they are explicitly stated by the clinician or patient.
11. Preserve negation, temporality, laterality, and numeric detail when explicitly present.

## Additional clinical adaptation rules
- Use the patient's name if explicitly available; otherwise use "The patient".
- Use "The clinician" for clinician findings, assessments, and plans.
- Keep each claim to one assertion only.
- Do not output claim labels, verdicts, explanations, or JSON.

## Example style
- The patient reports fatigue for the past two months.
- The patient denies blood in stool.
- The clinician suspects cervical radiculopathy.
- The clinician plans a cervical MRI.

## Output format
Return ONLY a bullet list of atomic claims.

<Input Transcript>
{transcript}
</Input Transcript>

'''.strip()



# ---------------------------------------------------------------------------
# V2 overrides: extraction prompts + severity-informed detection prompts
# ---------------------------------------------------------------------------

CAP_BASE_PROMPT_V3 = r"""
You are an expert clinical information extraction system.

Your task is to convert a clinician-patient dialogue transcript into dialogue-aware Clinical Atomic Propositions (CAPs).

[Definition of CAP]
A Clinical Atomic Proposition (CAP) is:
- one clinically meaningful assertion,
- standalone and self-contained,
- verifiable against transcript evidence,
- explicit about speaker and subject,
- and preserves clinically important modifiers such as negation, temporality, laterality, numeric value, severity, and management intent.

Only include VALID, VERIFIABLE declarative claims in `atomic_propositions`.

[Validity Rules - CRITICAL]
Do NOT include invalid / non-verifiable units as CAPs.
Instead, place them in `filtered_nonverifiable_units`.

Invalidity types:
- imperative: instruction / command / directive without a truth-evaluable factual claim
- interrogative: question
- incomplete: fragment without a complete assertion
- vague: unresolved subject, unresolved referent, or ambiguous clinical object that cannot be resolved from context

[Contextual Enrichment Rules - CRITICAL]
To prevent information loss in standalone propositions, you MUST:
1. Resolve deictic expressions and contextual references when possible.
   - Example: "this hand" -> "left hand" if grounded by the dialogue context.
2. Replace pronouns in `proposition_text` with explicit subjects.
3. Use the patient's actual name in patient-related `proposition_text` ONLY when it is explicitly recoverable from the current transcript chunk.
4. Keep the clinician as "The clinician" unless a named clinician identity is clinically relevant.
5. Preserve patient-reported symptom persistence, carry-over, progression, worsening, or change over time when explicitly stated.
6. Preserve explicit treatment plans, referrals, follow-up timing, return precautions, and patient-facing management instructions as high-priority propositions.

[Name Handling Rules - CRITICAL]
- The example JSON is illustrative only. NEVER copy or reuse patient names from the example output.
- Do NOT invent or guess a patient name from prior cases, prior chunks, memory, or outside context.
- This transcript may be chunked. The current chunk may not contain the patient's name.
- If the patient's name is not explicitly available in the current transcript chunk, use "The patient" in patient-related `proposition_text`.
- If the clinician is unnamed, use "The clinician".
- Never output a named patient unless that exact name is grounded in the current transcript chunk.
- If a proposition would otherwise contain an unsupported proper name, rewrite it with a generic grounded subject such as "The patient".
- Do NOT copy patient names from IE hints, prior chunks, prior examples, or previously processed encounters unless the current chunk itself explicitly contains that same name.

[CAP Schema]
Valid category values:
- Demographic
- ChiefComplaint
- History
- Allergy
- Symptom
- Finding
- Diagnosis
- MedicationPlan
- TestPlan
- FollowUpPlan
- UncertainOrNoise

Valid speaker values:
- patient
- clinician
- test_result
- unknown

Valid status values:
- affirmed
- negated
- uncertain

Valid temporality values:
- past
- current
- future
- relative

Valid predicate values:
- reports
- denies
- has
- shows
- diagnoses
- suspects
- prescribes
- orders
- recommends
- instructs
- plans
- notes

Valid claim_type_tags values:
- temporal
- quantitative
- causal
- medication
- plan
- diagnostic
- symptom
- exam
- safety_relevant

[Extraction Rules]
1. One CAP = exactly one clinical assertion.
2. Make speaker explicit.
3. Capture negation and uncertainty explicitly.
4. Preserve clinically important laterality, temporality, severity, and management intent inside `proposition_text`.
5. `proposition_text` must be a standalone English sentence.
6. Do not infer facts not explicitly grounded in the transcript.
7. Distinguish current findings from future plans.
8. Use `UncertainOrNoise` only for unclear, ASR-corrupted, or clinically unusable text that is still truth-evaluable enough to preserve.
9. Do not convert questions, tentative possibilities, or clinician speculation into affirmed current diagnoses or medications.
10. When a sentence contains one central clinical assertion plus low-value descriptive polish, preserve the central clinical assertion.
11. Return compact, flat JSON only. Omit null fields entirely. Do not include nested objects.
12. Return at most 12 atomic propositions for this transcript chunk. Prefer the most clinically salient propositions.
13. Treat each transcript chunk independently for identity grounding. If identity is missing in the chunk, fall back to generic grounded subjects such as "The patient" or "The clinician".
14. If the chunk starts mid-conversation and the subject is implicit, prefer safe generic subjects rather than unsupported named entities.

[Output Requirements]
Return valid JSON only.
Return one JSON object with this structure:
{{
  "atomic_propositions": [
    {{
      "prop_id": "P1",
      "category": "Symptom",
      "speaker": "patient",
      "predicate": "reports",
      "status": "affirmed",
      "temporality": "current",
      "claim_type_tags": ["symptom"],
      "proposition_text": "The patient reports pain in the left hand."
    }}
  ],
  "filtered_nonverifiable_units": ["Is the pain worse at night?"]
}}

- Include only these keys inside each atomic proposition:
  `prop_id`, `category`, `speaker`, `predicate`, `status`, `temporality`, `claim_type_tags`, `proposition_text`
- Omit any key whose value would be null or empty.
- `filtered_nonverifiable_units` must be a flat list of strings.
- Do not include explanations outside the JSON.
- Do NOT copy the example patient name into the output unless that exact name appears in the current transcript chunk.
- If the current chunk does not explicitly contain the patient's name, prefer "The patient" in all patient-related propositions.

<Input Transcript>
{transcript}
</Input Transcript>
""".strip()

CAP_KIWI_HINT_BLOCK_V3 = r"""
[Kiwi IE Hint Usage]

You are given auxiliary IE hints:

<IE_ENTITIES>
{entities}
</IE_ENTITIES>

<IE_RELATIONS>
{relations}
</IE_RELATIONS>

Use them ONLY for:
- span recovery
- modifier recovery
- recall improvement

Rules:
- Transcript is the single source of truth
- Do NOT output raw entities
- Do NOT trust mapping blindly
"""

CAP_KIWI_UMLS_BLOCK_V3 = r"""
[UMLS Mapping Usage]

You may use mapping candidates ONLY to:
- improve proposition wording when the transcript meaning is already clear

Rules:
- Prefer transcript wording
- Do not override meaning
"""


DETECT_PROMPT_HALLUCINATION_STRESS_V2 = r"""
You are a Senior Clinical Auditor checking for clinically meaningful hallucinations in a clinical summary.

You are given:
1. TranscriptEvents as indexed structured clinical events (TE0, TE1, ...)
2. SummaryEvents as indexed clinical events derived from summary atomic facts (SE0, SE1, ...)

TranscriptEvents are the source of truth.
SummaryEvents are the claims to verify.
Use strict grounding only.
Do not use outside medical knowledge to rescue unsupported content.
Event objects may include `event_type`, `event_slots`, and `asgari_alignment`; use them as grounding hints for meaning, not as independent evidence.

Each SummaryEvent may represent one or more summary atomic facts from the same underlying clinical event.
Do not decompose one SummaryEvent back into multiple hallucinations unless it contains clearly distinct clinically meaningful errors with different downstream consequences.

Canonical event types may include:
- SymptomState
- DiagnosisImpression
- MedicationState
- MedicationPlan
- FindingResult
- TestProcedurePlan
- FollowUpSafety
- PMFSContext

For each SummaryEvent, reason internally using this support taxonomy:
- Supported
- Partially supported
- Not addressed
- Contradicted / outdated by later facts
- Low-value mismatch / ignore

Definitions:
- Supported: the same clinical meaning is grounded by one or more TranscriptFacts, even if wording differs.
- Partially supported: most of the meaning is grounded, or the claim is reasonably recoverable by combining nearby TranscriptFacts, but some detail is less explicit.
- Not addressed: the transcript does not clearly contain the claim.
- Contradicted / outdated by later facts: the claim conflicts with the final grounded state or with a later correction.
- Low-value mismatch / ignore: the difference is stylistic, documentation-related, over-segmented, or too minor to matter clinically.

Decision rule:
- Do NOT return Supported, Partially supported, or Low-value mismatch / ignore items as hallucinations.
- A SummaryEvent that is merely Not addressed should NOT be returned by default.
- Return a SummaryEvent only if it is clinically meaningful and one of the following is true:
  - it introduces a new unsupported clinical fact,
  - it contradicts the final grounded state,
  - it changes the final medication or safety state,
  - it adds clinically consequential specificity that changes interpretation,
  - it fabricates or materially changes a plan, referral, order, procedure, or follow-up commitment.

Lack of direct lexical overlap is not enough.
If the claim is reasonably recoverable by combining nearby TranscriptFacts, prefer Supported or Partially supported rather than hallucination.
If the apparent mismatch could be explained by extraction granularity noise, over-segmentation, bundled evidence, schema mismatch, or harmless abstraction, do NOT return it as a hallucination.

Your goal is not to list every unsupported atomic mismatch.
Your goal is to identify the smallest set of clinically central hallucination events that best explains the materially unsafe or meaning-changing unsupported content in the summary.

Apply a severity-informed clinical importance prior.
High-priority hallucination candidates are those that could change diagnosis, differential interpretation, management, medication safety, procedure eligibility, follow-up, or patient safety understanding.

Strongly prioritize hallucinations involving:
- new symptom / diagnosis / medication / exposure not grounded in the transcript
- wrong final medication state or safety-critical exposure state
- wrong abnormal result / numeric value / severity / laterality / temporality when clinically consequential
- fabricated plan / referral / procedure / follow-up
- clinically meaningful contradiction of the final grounded state
- unsupported active symptom burden or progression tied to the main presenting problem

For stress-benchmark alignment, pay special attention to these hallucination families:
- Fabricated fact or fabricated negation: a symptom, diagnosis, medication, result, plan, referral, or communication detail that was never stated
- Negation distortion: the summary denies or removes a clinically relevant event or symptom that was actually present
- Context conflation: the summary mixes details across speakers, timelines, settings, or sections in a way that changes meaning
- Assumed causality: the summary invents a causal or explanatory link that was not explicitly grounded in the transcript

Use these families as prioritization guidance, not as a license to over-flag minor wording differences.

Treat the following as low-priority and usually do NOT return them unless they independently change diagnosis, medication safety, procedural eligibility, or the final management decision:
- exact provider names, credentials, or authorship details
- exact formulation words such as tablet/capsule/injection form when the medication itself is already grounded
- minor temporal polishing or storytelling detail
- routine negative exam or review-of-systems template completions
- extra physical exam granularity not linked to a central diagnostic or procedural decision
- wording that makes a plan sound slightly more scheduled or slightly more concrete without changing the actual intended test/procedure/treatment
- date normalization or formatting recovery unless the exact date itself is clinically important
- low-value local detail that is recoverable by combining nearby transcript facts

If a low-priority unsupported detail appears in the same local summary region as a higher-priority fabricated symptom, diagnosis, medication-state error, plan error, or causality error, ignore the low-priority detail and return only the higher-priority event.
Do not let attribution detail, temporal polish, minor exam detail, or documentation style differences inflate the hallucination count when a more clinically central unsupported event is already present nearby.

Equivalent clinical abstraction is supported unless it changes certainty, negation, severity, laterality, temporality, medication state, or management.

Apply SpecificityInflation only when the added specificity changes clinical interpretation in a meaningful way.
Do NOT use SpecificityInflation for harmless sharpening of wording, stylistic completion, or minor detail enrichment.
Use it only when the added specificity changes:
- certainty
- diagnosis identity
- laterality
- severity
- duration in a clinically consequential way
- medication exposure state
- test/procedure eligibility
- treatment commitment

Counting rule:
- Count hallucinations at the event level, not at the raw atomic-fact level.
- If a summary sentence contains one supported scaffold plus one unsupported added clause, return only the clinically salient unsupported SummaryEvent rather than every mechanically derived conjunct.
- Multiple unsupported atomic claims arising from the same summary sentence should be treated as one hallucination event unless they represent clearly distinct clinically meaningful errors with different downstream consequences.
- If several summary atomic facts reflect the same underlying medication state, diagnosis, finding, or plan error, they should already be represented as one SummaryEvent. Return that one SummaryEvent ID only.
- If several unsupported SummaryEvents arise from the same local summary region, return only the single most clinically central hallucination event unless they represent clearly separate clinical errors with different downstream consequences.
- If one underlying issue produced several atomic summary facts before event consolidation, count that as one hallucination event.

When in doubt, prefer under-clustering low-value mismatches into one central hallucination event rather than over-counting multiple atomic mismatches from the same sentence or adjacent summary region.
The returned set should be the smallest clinically sufficient explanation of the summary's unsupported content.
Do not inflate hallucination counts because the extraction layer produced many fine-grained atomic claims from one sentence. Count the central clinical error event, not every derived atom.

For synthetic benchmark alignment, prefer unsupported additions that introduce a new symptom, diagnosis, medication, exposure state, exam finding, or management fact that was not present in the transcript at all.
Prefer these over low-value attribution differences, documentation style differences, or local narrative embellishments.
Prefer the most salient injected-style unsupported event from a sentence over secondary low-value mismatches in that same sentence.

Taxonomy tags you may use in reasons:
- Fabrication
- SpecificityInflation
- AttributionError
- TemporalDistortion
- StateContradiction
- OutdatedState
- MedicationState
- PlanMismatch
- DiagnosticMismatch
- FindingMismatch
- HighPriority
- LowPriorityIgnored

Return strict JSON only.
Important:
- In `unsupported_summary_event_ids`, return the indexed SummaryEvent IDs exactly as `SE0`, `SE1`, `SE2`, ...
- Do NOT return `S0` / `SF1` / inner fact_id strings there.
- Each reason should begin with its matching `SE#` ID.
- Each reason should describe the central unsupported event, not every tiny derivative mismatch.
- `total_hallucination_count` must equal the number of unique IDs returned.

{{
  "unsupported_summary_event_ids": ["SE2", "SE7"],
  "hallucination_reasons": [
    "SE2 [MedicationState]: prednisone is not confirmed anywhere in the transcript events.",
    "SE7 [OutdatedState]: the summary event denies anticoagulant use despite a later Eliquis-confirmed final state."
  ],
  "total_hallucination_count": 2
}}

<TranscriptEvents>
{indexed_facts}
</TranscriptEvents>

<SummaryEvents>
{indexed_summary_facts}
</SummaryEvents>
""".strip()


DETECT_PROMPT_OMISSION_STRESS_V2 = r"""
You are a Senior Clinical Auditor checking for omissions in a clinical summary.

You are given:
1. TranscriptEvents as indexed structured clinical events (TE0, TE1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptEvents are the source of truth.
Judge omissions from TranscriptEvents only.
The summary does NOT need to repeat facts verbatim.
An event is omitted only when its clinically relevant meaning is not recoverable anywhere in the summary.

Each TranscriptEvent may represent one or more transcript facts that describe the same underlying clinical event.
Return omissions at the event level, not at the raw fact level.
Do not split one TranscriptEvent into multiple omissions unless it truly contains distinct missing clinical gaps with different downstream consequences.

This detector serves two downstream evaluation axes at once:
1. injection-faithful omission recovery
2. clinical-priority omission auditing

So your job is NOT only to find high-severity omissions.
You should also return clearly missing Current Issues / Information and Plan / PMFS events when they are explicitly present in the transcript event and genuinely absent from the summary, even if they are minor.

Use this internal decision ladder for EACH TranscriptEvent:
Step 1. Recoverability gate
- Is the core meaning already present anywhere in the summary?
- If YES, do NOT return it as omission.
- Recoverable means the summary may paraphrase, compress, abstract, or merge the event, as long as the core clinical meaning is still there.
- A detail is still recoverable if the summary expresses the same event at a higher level.
- Do NOT require exact wording, exact section placement, or exact sentence alignment.

Step 2. Contradiction gate
- If the summary states the opposite or an incompatible claim for the same event, do NOT mark omission.
- That belongs under hallucination / contradiction, not omission.

Step 3. Missingness gate
- If the event is not recoverable anywhere in the summary, decide whether it is:
  - Missing and clinically important
  - Missing but minor / lower-priority

Return BOTH types when they are genuinely missing.
However, if several TranscriptEvents are just over-segmented support details of the same missing event, return only the most representative one.

Important: do NOT let clinical-priority bias hide a true injected omission.
If a transcript event is clearly present in the transcript and clearly absent from the summary, it can still be a valid omission even when it is not the most safety-critical omission in the case.

Recoverability examples:
- If the transcript says the patient is doing well postoperatively, and the summary says the patient is doing well and progressing with home exercises, that is usually recoverable, so do NOT return omission.
- If the transcript says vitals look good and blood pressure / heart rate are normal, and the summary keeps only temperature normal, the normal blood pressure / heart rate event is NOT recoverable and may be omitted.
- If the transcript says follow-up in 1-2 weeks and discuss lithotripsy if not improved, and the summary includes that same follow-up plan in different words, do NOT return omission.

Prioritize event families using the following taxonomy, but do not use it as a hard filter:
- CurrentIssueMissing: active symptoms, symptom burden, persistence, progression, important negatives, active current issues
- InformationPlanMissing: assessment, treatment, test, referral, follow-up, return precautions, safety-net, patient instructions
- PMFSMissing: past medical history, medications, allergies, family/social history when relevant to current diagnosis or management
- KeyResultValueMissing: important findings, abnormal or safety-relevant results, quantitative values

Clinical-priority guidance:
- High priority usually includes treatment plans, medication instructions, referrals, procedure planning, follow-up timing, safety instructions, key results, active main-problem symptoms, and final clinically consequential states.
- Lower priority may include isolated background details, low-value specificity, minor exam granularity, or contextual phrasing.
- But lower priority does NOT mean “not an omission.” If the event is clearly present in the transcript and clearly absent from the summary, you may still return it.

Over-segmentation control:
- If several transcript facts describe the same missing symptom state, plan, medication instruction, or finding bundle, return only the single most representative TranscriptEvent.
- Prefer the clinically central event over subordinate rationale or support detail.
- Do not return structural labels, list numbering, clinician questions, clarification turns, conversational scaffolding, or social filler.

History handling:
- Treat History events conservatively.
- Return them when they are clearly relevant or when they correspond to a direct synthetic omission target that is truly absent from the summary.

Do NOT return omission labels for:
- information already recoverable elsewhere in the summary
- exact-wording mismatches only
- local section-placement differences only
- contradictions or incompatible opposite claims
- tiny support details when a larger parent event is already present in the summary

Taxonomy tags you may use in reasons:
- CurrentIssueMissing
- InformationPlanMissing
- PMFSMissing
- KeyResultValueMissing
- SafetyCritical
- FinalState
- ClinicalPriority

Return strict JSON only.
Important:
- In `omitted_event_ids`, return the indexed TranscriptEvent IDs exactly as `TE0`, `TE1`, `TE2`, ...
- Each reason should begin with its matching `TE\#` ID.
- `total_omission_count` must equal the number of unique IDs returned.
- Be conservative about semantic recoverability, but not so conservative that clearly removed injected events disappear.

{{
  "omitted_event_ids": ["TE19", "TE24"],
  "omission_reasons": [
    "TE19 [InformationPlanMissing][ClinicalPriority]: follow-up instruction is clearly absent from the summary.",
    "TE24 [CurrentIssueMissing]: normal blood pressure and heart rate are stated in the transcript event but not recoverable from the summary."
  ],
  "total_omission_count": 2
}}

<TranscriptEvents>
{indexed_facts}
</TranscriptEvents>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
""".strip()


DETECT_PROMPT_OMISSION_STRESS_V3_S_ONLY = r"""
You are a Senior Clinical Auditor checking for omissions in the Subjective portion of a clinical summary.

You are given:
1. TranscriptEvents as indexed structured clinical events (TE0, TE1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptEvents are the source of truth.
Judge omissions from TranscriptEvents only.
The summary does NOT need to repeat facts verbatim.
An event is omitted only when its clinically relevant meaning is not recoverable anywhere in the summary.

This detector serves two goals at once:
1. injection-faithful omission recovery
2. clinical-priority omission auditing

Your FIRST priority is to decide whether each transcript event under review is itself genuinely missing.
Do NOT replace a clearly missing transcript event with some other omission that merely seems more important.

This is an S-only SOAP audit.
Only consider events that belong in Subjective content, such as:
- chief complaint
- HPI symptoms and symptom course
- subjective ROS
- patient-reported PMFS context
- patient-reported medication use/adherence relevant to the complaint

Do NOT treat the following as S-only omission targets:
- physical exam findings
- objective vitals
- lab or imaging results
- clinician assessment/impression unless clearly framed as patient-reported history
- treatment plans, orders, referrals, procedures, follow-up instructions, or safety-net guidance

Event objects may include `event_type`, `event_slots`, and `asgari_alignment`; use them as grounding hints for meaning and taxonomy alignment.

For each TranscriptEvent, apply this internal ladder:
Step 1. S-scope gate
- If the event is not appropriate for the Subjective section, do NOT return it.

Step 2. Recoverability gate
- If the event's core meaning is already present anywhere in the summary, do NOT return it.
- Recoverable means the summary may paraphrase, compress, abstract, or merge the event, as long as the core clinical meaning is still there.
- Do NOT require exact wording, exact sentence alignment, or exact section placement.

Step 3. Contradiction gate
- If the summary states the opposite or an incompatible claim for the same event, do NOT mark omission.
- That belongs under hallucination / contradiction, not omission.

Step 4. Missingness gate
- If the event is within S-scope and not recoverable anywhere in the summary, return it as omitted.
- If several TranscriptEvents are just over-segmented support details of the same missing subjective event, return only the most representative one.
- If the transcript event is explicit, genuinely absent, and appropriate for Subjective content, return that event even if another omission elsewhere in the case is more clinically important.

Local fidelity rule:
- Prefer the omission that matches the specific symptom/history/current-issue trajectory expressed by the transcript event under review.
- Do NOT substitute a different plan/result/follow-up omission when the current transcript event itself is clearly missing.
- Minor Current Issues omissions are still valid omissions if they are explicit and not recoverable.

Taxonomy guidance:
- CurrentIssueMissing: active symptoms, symptom burden, persistence, progression, important negatives, active current issues
- PMFSMissing: past medical history, medications, allergies, family/social history when relevant to the current complaint
- ClinicalPriority: use when the missing subjective event materially changes clinical understanding

Important:
- minor current issues can still be valid omissions if clearly present in the transcript and absent from the summary
- do NOT over-count multiple TranscriptEvents from the same local symptom/state bundle
- do NOT return structural labels, clinician questions, clarification turns, or conversational filler
- do NOT ignore a valid missing local event just because a different omission elsewhere in the case is more safety-critical
- when unsure, prefer the smaller, explicit missing transcript event over a broader inferred omission not directly aligned to the transcript event being judged

Return strict JSON only.
Important:
- In `omitted_event_ids`, return the indexed TranscriptEvent IDs exactly as `TE0`, `TE1`, `TE2`, ...
- Each reason should begin with its matching `TE#` ID.
- `total_omission_count` must equal the number of unique IDs returned.

{
  "omitted_event_ids": ["TE4", "TE9"],
  "omission_reasons": [
    "TE4 [CurrentIssueMissing]: the patient's active symptom detail is present in the transcript event but not recoverable from the summary.",
    "TE9 [PMFSMissing]: relevant patient-reported history is absent from the summary."
  ],
  "total_omission_count": 2
}

<TranscriptEvents>
{indexed_facts}
</TranscriptEvents>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
""".strip()


DETECT_PROMPT_OMISSION_STRESS_V4_MAJOR_EVENT_VERIFY = r"""
You are a Senior Clinical Auditor verifying MAJOR omissions in a clinical summary.

You are given:
1. TranscriptEvents as indexed structured clinical events (TE0, TE1, ...)
2. SummarySentences as indexed body sentences from the summary (S0, S1, ...)

TranscriptEvents are the source of truth.
This is a verification task, not an open-ended search task.
You must verify each transcript event against the summary and decide whether that same event is covered, contradicted, minorly missing, or majorly missing.

Event objects may include `event_type`, `clinical_priority`, `severity`, `event_slots`, and `asgari_alignment`; use them as grounding hints.

Definition of MAJOR omission:
- A major omission is missing information that could change diagnosis, differential interpretation, patient management, medication safety, test/procedure eligibility, follow-up, return precautions, or overall safety understanding.
- Think in Asgari-style safety terms: a major omission can create diagnostic delay, management distortion, or safety-net failure.
- Think in MED-OMIT-style importance terms: if the omitted information would materially change the diagnostic picture or management plan, treat it as major.

Verification labels:
- Covered: the event's core meaning is recoverable somewhere in the summary
- Contradicted: the summary states an incompatible opposite claim for the same event
- MissingMinor: the event is not recoverable, but the gap is clinically minor
- MissingMajor: the event is not recoverable and the gap is clinically major
- Ignore: low-value, redundant, or non-central support detail that should not be surfaced on its own

Major omission criteria:
- diagnosis / impression that changes clinical interpretation
- medication initiation / discontinuation / dose change / adherence state
- major findings or quantitative values that change interpretation
- tests, procedures, referrals, or workup steps
- follow-up timing, safety-net, or return precautions
- main symptom burden, progression, or clinically meaningful important negatives

Rules:
1. Verify each TranscriptEvent internally before deciding.
2. Do NOT replace one missing event with a different omission elsewhere in the case.
3. Do NOT require exact wording; paraphrase and higher-level summary are allowed if the same event is recoverable.
4. If the summary expresses the same event at a higher level, prefer Covered.
5. If part of an event is omitted and that omitted part could itself change diagnosis or management, you may still mark the event as MissingMajor.
   - Example: omission of a critical value, important negative, laterality, or follow-up timing can still be major even when the broader topic appears in the summary.
6. If several transcript events are support details for the same larger missing major event, you may mark support details as Ignore and keep the most central event as MissingMajor.
7. Return only MAJOR missing events in the ranked output.
8. Rank missing major events from most clinically central to less central.

Return strict JSON only.

{{
  "event_verifications": [
    {{
      "event_id": "TE0",
      "verdict": "Covered",
      "reason_tag": "Covered",
      "reason": "The same clinical event is recoverable in the summary."
    }},
    {{
      "event_id": "TE4",
      "verdict": "MissingMajor",
      "reason_tag": "MedicationPlanMissing",
      "reason": "The medication change is explicit in the transcript and not recoverable in the summary."
    }}
  ],
  "ranked_missing_major_event_ids": ["TE4"],
  "omission_reasons": [
    "TE4 [MedicationPlanMissing]: the medication change is explicit in the transcript and absent from the summary."
  ],
  "total_omission_count": 1
}}

Important:
- `ranked_missing_major_event_ids` must contain only `TE#` IDs with verdict `MissingMajor`
- `total_omission_count` must equal the number of unique IDs in `ranked_missing_major_event_ids`
- Do not return MissingMinor or Covered events in `ranked_missing_major_event_ids`

<TranscriptEvents>
{indexed_facts}
</TranscriptEvents>

<SummarySentences>
{indexed_summary_sentences}
</SummarySentences>
""".strip()



# METHODS_V3
METHODS = {
    "A": {
        "name": "Method A: BN-style Atomic Facts",
        "extract_prompt": EXTRACT_BN_PROMPT,
        "summary_extract_prompt": EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT,
        "detect_prompt_normal": DETECT_PROMPT_NORMAL,
        "detect_prompt_stress": DETECT_PROMPT_STRESS,
        "detect_prompt_hallucination_stress": DETECT_PROMPT_HALLUCINATION_STRESS,
        "detect_prompt_omission_stress": DETECT_PROMPT_OMISSION_STRESS,
        "output_mode": "text",
        "source_columns": ['transcript'],
        "fact_column_candidates": ['transcript_bn_facts', 'transcript_atomic_facts', 'transcript_a_facts'],
    },
    "B": {
        "name": "Method B: VeriFact-style Atomic Claims",
        "extract_prompt": EXTRACT_VERIFACT_PROMPT,
        "summary_extract_prompt": EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT,
        "detect_prompt_normal": DETECT_PROMPT_NORMAL,
        "detect_prompt_stress": DETECT_PROMPT_STRESS,
        "detect_prompt_hallucination_stress": DETECT_PROMPT_HALLUCINATION_STRESS,
        "detect_prompt_omission_stress": DETECT_PROMPT_OMISSION_STRESS,
        "output_mode": "text",
        "source_columns": ['transcript'],
        "fact_column_candidates": ['transcript_verifact_claims', 'transcript_atomic_claims', 'transcript_b_facts'],
    },
    "C": {
        "name": "Method C: Ours-only Clinical Atomic Propositions",
        "extract_prompt": CAP_BASE_PROMPT_V3,
        "summary_extract_prompt": EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT,
        "detect_prompt_normal": DETECT_PROMPT_NORMAL,
        "detect_prompt_stress": DETECT_PROMPT_STRESS,
        "detect_prompt_hallucination_stress": DETECT_PROMPT_HALLUCINATION_STRESS,
        "detect_prompt_omission_stress": DETECT_PROMPT_OMISSION_STRESS,
        "output_mode": "json",
        "source_columns": ['transcript'],
        "fact_column_candidates": ['transcript_cap_facts', 'transcript_ours_cap_facts', 'transcript_c_facts'],
    },
    "D": {
        "name": "Method D: Ours+Kiwi Clinical Atomic Propositions (No Mapping)",
        "extract_prompt": CAP_BASE_PROMPT_V3 + CAP_KIWI_HINT_BLOCK_V3,
        "summary_extract_prompt": EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT,
        "detect_prompt_normal": DETECT_PROMPT_NORMAL,
        "detect_prompt_stress": DETECT_PROMPT_STRESS,
        "detect_prompt_hallucination_stress": DETECT_PROMPT_HALLUCINATION_STRESS,
        "detect_prompt_omission_stress": DETECT_PROMPT_OMISSION_STRESS,
        "output_mode": "json",
        "source_columns": ['transcript', 'entities', 'relations'],
        "fact_column_candidates": ['transcript_cap_kiwi_nomap_facts', 'transcript_ours_cap_kiwi_nomap_facts', 'transcript_d_facts'],
    },
    "E": {
        "name": "Method E: Ours+Kiwi Clinical Atomic Propositions (With UMLS Mapping)",
        "extract_prompt": CAP_BASE_PROMPT_V3 + CAP_KIWI_HINT_BLOCK_V3 + CAP_KIWI_UMLS_BLOCK_V3,
        "summary_extract_prompt": EXTRACT_SUMMARY_ATOMIC_FACTS_PROMPT,
        "detect_prompt_normal": DETECT_PROMPT_NORMAL,
        "detect_prompt_stress": DETECT_PROMPT_STRESS,
        "detect_prompt_hallucination_stress": DETECT_PROMPT_HALLUCINATION_STRESS,
        "detect_prompt_omission_stress": DETECT_PROMPT_OMISSION_STRESS,
        "output_mode": "json",
        "source_columns": ['transcript', 'entities', 'relations'],
        "fact_column_candidates": ['transcript_cap_kiwi_umls_facts', 'transcript_ours_cap_kiwi_umls_facts', 'transcript_e_facts'],
    },
}


TAB_TO_METHOD = {cfg["name"]: key for key, cfg in METHODS.items()}
METHOD_NAME_TO_KEY = {cfg["name"]: key for key, cfg in METHODS.items()}

def get_extract_prompt(method_key: str) -> str:
    return METHODS[method_key]["extract_prompt"]

def get_summary_extract_prompt(method_key: str) -> str:
    return METHODS[method_key]["summary_extract_prompt"]

def get_detect_prompt(method_key: str, dataset_type: str) -> str:
    if dataset_type.lower() == "stress":
        return METHODS[method_key]["detect_prompt_stress"]
    return METHODS[method_key]["detect_prompt_normal"]

def get_detect_hallucination_prompt(method_key: str, dataset_type: str) -> str:
    if dataset_type.lower() == "stress":
        return METHODS[method_key]["detect_prompt_hallucination_stress"]
    return METHODS[method_key]["detect_prompt_normal"]

def get_detect_omission_prompt(method_key: str, dataset_type: str) -> str:
    if dataset_type.lower() == "stress":
        return METHODS[method_key]["detect_prompt_omission_stress"]
    return METHODS[method_key]["detect_prompt_normal"]

def method_display_names() -> list[str]:
    return [cfg["name"] for cfg in METHODS.values()]

def clone_methods() -> dict:
    return deepcopy(METHODS)


for _mk in METHODS:
    METHODS[_mk]["detect_prompt_hallucination_stress"] = DETECT_PROMPT_HALLUCINATION_STRESS_V2
    METHODS[_mk]["detect_prompt_omission_stress_backup"] = DETECT_PROMPT_OMISSION_STRESS_V2
    METHODS[_mk]["detect_prompt_omission_stress_s_only"] = DETECT_PROMPT_OMISSION_STRESS_V3_S_ONLY
    METHODS[_mk]["detect_prompt_omission_stress"] = DETECT_PROMPT_OMISSION_STRESS_V4_MAJOR_EVENT_VERIFY
