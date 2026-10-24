"""
treatment.py — BEI Treatment Synthesis via Claude Sonnet
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Hard clinical rules (v1.1 — 5 rules):
  Rule 1: U:Cr ratio >150 → CATABOLIC HYPERUREAEMIA (not AKI). Lead with catabolism.
  Rule 2: IL-10 Energy >0.70 → Corticosteroids CONTRAINDICATED (Section E).
  Rule 3: D-dimer alone does NOT establish DIC — require ISTH DIC score ≥5.
  Rule 4: Never cite 'Urea <20 mmol/L' as CRRT target — use KDIGO/RENAL/SSC 2021.
  Rule 5: If CATABOLIC HYPERUREAEMIA → protein ≥1.5 g/kg/day (NOT 0.8–1.2).
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
import httpx

from rag_retrieval import format_chunks_for_claude
from config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, INVESTIGATIONAL_DISCLAIMER

logger = logging.getLogger(__name__)


def _build_treatment_prompt(
    data: dict,
    chunks: List[dict],
) -> str:
    """
    Build the full treatment synthesis prompt.
    Hard rules are computed from this patient's biomarker values — not generic.
    """
    bei     = data["bei"]
    raw     = data["raw_input"]
    pid     = data.get("patient_id", "UNKNOWN")
    tipping = bei.get("tipping_marker", "?")
    tier    = bei.get("tipping_tier", "?")
    zone    = bei.get("bei_zone", "?")
    tc      = bei.get("tipping_contributions", {})
    ep      = bei.get("energy_profile", {})
    lit     = format_chunks_for_claude(chunks)

    neg_tcs = sorted(
        [(m, v) for m, v in tc.items() if v < 0], key=lambda x: x[1]
    )[:5]

    neg_tc_str = "\n".join(
        f"  {m}: TC={v:+.5f}, Energy={ep.get(m, 0):.4f}" for m, v in neg_tcs
    )
    raw_str = "\n".join(f"  {k}: {v}" for k, v in raw.items())
    corrections = "\n".join(f"  {c}" for c in data.get("unit_corrections", []))

    gcs_note = (
        "GCS ALARM ACTIVE: GCS ≤14 detected. 67% 90-day mortality threshold."
        if bei.get("gcs_alarm")
        else "GCS is normal (>14). No neurological alarm."
    )
    lactate_note = (
        "LACTATE IS MISSING. First intervention must be immediate Lactate "
        "measurement. BEI score may be artificially inflated without it. "
        "True zone classification uncertain until Lactate is known."
        if bei.get("lactate_missing_warning")
        else "Lactate present in dataset."
    )

    # ── HARD RULE COMPUTATION (patient-specific) ───────────────────────────
    # Rule 1 + 5: U:Cr ratio determines catabolism vs AKI and protein target
    urea_val   = float(raw.get("Urea", 0) or 0)
    creat_val  = float(raw.get("Creatinine", 1) or 1)
    creat_si   = creat_val / 1000.0          # μmol/L → mmol/L
    uc_ratio   = round(urea_val / creat_si, 1) if creat_si > 0 else 0
    uc_mechanism = "CATABOLIC HYPERUREAEMIA" if uc_ratio > 150 else "AKI-PREDOMINANT"

    rule1_text = (
        "CATABOLISM IS THE PRIMARY MECHANISM. Lead Section A with this. "
        "Sepsis-driven skeletal muscle proteolysis is producing excess urea, "
        "NOT kidney failure. The creatinine elevation is mild and secondary. "
        "Primary intervention = protein ≥1.5 g/kg/day + early enteral nutrition. "
        "Do NOT frame as AKI-first."
        if uc_ratio > 150
        else "AKI framing is appropriate for this ratio. Proceed with renal-first approach."
    )

    # Rule 2: IL-10 energy → steroid contraindication
    il10_energy  = ep.get("IL-10", 0.0)
    steroid_status = "CONTRAINDICATED" if il10_energy > 0.70 else "ASSESS INDIVIDUALLY"
    steroid_rule = (
        f"EXPLICITLY FLAG in Section E: Corticosteroids are CONTRAINDICATED. "
        f"IL-10 Energy={il10_energy:.4f} indicates intact anti-inflammatory reserve. "
        f"Steroid administration suppresses appropriate immune response in a "
        f"non-immunoparalysed patient. This increases infection risk."
        if il10_energy > 0.70
        else "Corticosteroid candidacy requires individual haemodynamic assessment."
    )

    # Rule 3: DIC criteria
    ddimer_val = float(raw.get("D-dimer", 0) or 0)
    plt_val    = float(raw.get("Platelets", 0) or 0)
    dic_evidence = (
        f"D-dimer={ddimer_val} ng/mL, Platelets={plt_val} ×10⁹/L. "
        f"{'Platelets normal — DIC criteria NOT met on current data.' if plt_val > 150 else 'Platelets low — consider full ISTH DIC score.'}"
    )

    # Rule 4: CRRT indications
    crrt_indications = (
        f"Urea={urea_val} mmol/L, Creatinine={creat_val} μmol/L. "
        f"CRRT indication requires ONE of: refractory fluid overload, "
        f"pH <7.15, K⁺ >6.5 mmol/L, uraemic complications, or haemodynamic "
        f"instability refractory to vasopressors. Urea level alone is NOT "
        f"an indication. Do NOT cite 'Urea <20 mmol/L' as a CRRT target — "
        f"this threshold does not exist in KDIGO or RENAL trial guidelines."
    )

    # Rule 5: Protein target
    protein_rule = (
        f"U:Cr ratio={uc_ratio} — mechanism is {uc_mechanism}. "
        + (
            "Protein target = ≥1.5 g/kg/day (ESPEN 2019, Heyland 2011). "
            "Do NOT recommend 0.8–1.2 g/kg/day — protein restriction in "
            "catabolic sepsis accelerates muscle wasting and INCREASES mortality."
            if uc_ratio > 150
            else "Standard protein target 1.2–1.5 g/kg/day — monitor renal tolerance."
        )
    )

    sec_markers = ", ".join(m for m, _ in neg_tcs[1:4])

    return f"""You are a clinical decision-support AI for a sepsis ICU patient.
This is part of an FDA De Novo submission for SepsisAI BEI Module v4
(Adnexus Biotechnologies Inc). This is INVESTIGATIONAL — not FDA cleared.

You have been provided with:
1. A patient BEI (Biological Energy Index) output identifying the primary
   rate-limiting biological failure dimension
2. Peer-reviewed literature chunks retrieved specifically for this
   tipping dimension from the SepsisAI knowledge base
3. Five hard clinical rules computed from this patient's biomarker values

Your task is to synthesise a treatment protocol grounded in the provided
literature. Where the literature supports a recommendation, cite it.
Where it does not, say so explicitly rather than inventing support.

═══ PATIENT DATA ═══════════════════════════════════════
Patient: {pid}
BEI Score: {bei.get('bei_score', 0):.4f} [{zone} zone]
Zone detail: {bei.get('bei_zone_detail', '')}

PRIMARY TIPPING DIMENSION: {tipping} [{tier}-tier]
TC({tipping}): {tc.get(tipping, 0):+.5f}
  TC = W(marker) x [Energy(patient) - Energy(cohort mean)]
  Most negative TC = system failing fastest relative to cohort.

TOP FAILING DIMENSIONS:
{neg_tc_str}

RAW BIOMARKER VALUES:
{raw_str}

UNIT CORRECTIONS APPLIED:
{corrections if corrections else "None"}

GH5 sub-score: {bei.get('gh5_score', 'N/A')}
{gcs_note}
{lactate_note}
═══════════════════════════════════════════════════════

═══ HARD CLINICAL RULES — NON-NEGOTIABLE ════════════════
These rules are computed from THIS patient's values.
They override retrieved literature where in conflict.
Violating any rule invalidates this document for FDA submission.

RULE 1 — UREA MECHANISM CLASSIFICATION:
  Urea        = {urea_val} mmol/L
  Creatinine  = {creat_val} μmol/L
  U:Cr ratio  = {uc_ratio} (SI: Urea mmol/L ÷ Creatinine mmol/L)
  Threshold   = 150  (>150 = catabolism-dominant, not primary AKI)
  MECHANISM   = {uc_mechanism}
  {rule1_text}

RULE 2 — CORTICOSTEROID CONTRAINDICATION:
  IL-10 Energy = {il10_energy:.4f}
  Status       = {steroid_status}
  {steroid_rule}

RULE 3 — DIC THRESHOLD:
  {dic_evidence}
  D-dimer alone does NOT establish DIC.
  Do NOT recommend anticoagulation or invoke DIC pathway without
  ISTH DIC score ≥5 (requires platelet trend + PT/aPTT + fibrinogen
  + clinical context, not D-dimer alone).
  Frame D-dimer elevation as: "monitor coagulation profile; full
  ISTH DIC score required before anticoagulation decision."

RULE 4 — CRRT INDICATION:
  {crrt_indications}

RULE 5 — PROTEIN TARGET:
  {protein_rule}
═════════════════════════════════════════════════════════

═══ RETRIEVED LITERATURE (your grounding source) ═══════
{lit}
═══════════════════════════════════════════════════════

INSTRUCTIONS:
- Apply all 5 hard rules above — they are non-negotiable
- Ground every recommendation in the literature where possible
- If literature is insufficient, note the gap explicitly (Section F)
- Be specific to THIS patient's values — not generic sepsis
- Frame interventions as "consider", "assess candidacy for", "review"
- This is decision-support only — clinician makes final call
- Do NOT mention outcome_90d
- No preamble. Start immediately with SECTION A.

Produce EXACTLY these sections:

SECTION A — BIOLOGICAL PATHWAY SUMMARY
What specific biological failure pathway does {tipping} as the primary tipping
dimension represent for THIS patient? Apply Rule 1 — state the U:Cr ratio
and declare the mechanism (catabolism vs AKI) explicitly.
2-3 sentences. Cite literature where relevant.

SECTION B — IMMEDIATE INTERVENTIONS (0-6 hours)
3-5 specific interventions targeting the {tipping} pathway.
Apply Rule 5 for protein target if catabolism-dominant.
Format each as:
  Intervention | Mechanistic rationale specific to this patient | Target value | [citation]

SECTION C — TARGETED INTERVENTIONS (6-72 hours)
2-4 interventions targeting the secondary failing dimensions
({sec_markers}).
Apply Rule 3 for D-dimer. Apply Rule 4 for CRRT.
Same format as Section B.

SECTION D — MONITORING PROTOCOL
5 specific monitoring parameters with frequency and evidence-based
target values. Apply Rule 4 CRRT targets. Cite [N].

SECTION E — CONTRAINDICATION FLAGS
2-3 specific contraindication considerations.
Apply Rule 2 — state corticosteroid status explicitly.
Cite [N] where possible.

SECTION F — BIBLIOGRAPHY GAP ASSESSMENT
Which interventions above are NOT supported by the retrieved literature?
What papers would strengthen the evidence base for this tipping dimension?
This field feeds directly into the RAG pipeline bibliography update.

SECTION A — BIOLOGICAL PATHWAY SUMMARY"""


def synthesize_treatment(
    data: dict,
    chunks: List[dict],
    api_key: str,
) -> dict:
    """
    Call Claude Sonnet for BEI-guided treatment synthesis.

    Returns dict:
      {
        synthesis_text:     str,
        hard_rules_applied: List[str],
        model_used:         str,
        rag_chunk_count:    int,
        uc_ratio:           float,
        uc_mechanism:       str,
        il10_energy:        float,
        steroid_status:     str,
      }
    """
    bei   = data["bei"]
    raw   = data["raw_input"]
    ep    = bei.get("energy_profile", {})

    # Pre-compute hard rule values for response metadata
    urea_val  = float(raw.get("Urea", 0) or 0)
    creat_val = float(raw.get("Creatinine", 1) or 1)
    creat_si  = creat_val / 1000.0
    uc_ratio  = round(urea_val / creat_si, 1) if creat_si > 0 else 0
    uc_mechanism  = "CATABOLIC HYPERUREAEMIA" if uc_ratio > 150 else "AKI-PREDOMINANT"
    il10_energy   = ep.get("IL-10", 0.0)
    steroid_status = "CONTRAINDICATED" if il10_energy > 0.70 else "ASSESS INDIVIDUALLY"

    hard_rules_applied = [
        f"Rule 1: U:Cr={uc_ratio} → {uc_mechanism}",
        f"Rule 2: IL-10 Energy={il10_energy:.4f} → Corticosteroids {steroid_status}",
        "Rule 3: DIC requires ISTH score ≥5, not D-dimer alone",
        "Rule 4: CRRT indication per KDIGO (not Urea threshold)",
        f"Rule 5: Protein target {'≥1.5 g/kg/day' if uc_ratio > 150 else '1.2–1.5 g/kg/day'}",
    ]

    prompt = _build_treatment_prompt(data, chunks)

    # Verify all 5 rules injected
    rules_present = all(f"RULE {n}" in prompt for n in range(1, 6))
    if not rules_present:
        logger.error("Hard rules NOT found in prompt — aborting to prevent hallucination")
        raise RuntimeError("Hard rules injection failed")

    logger.info(
        f"Claude synthesis: patient={data.get('patient_id','?')}, "
        f"tipping={bei.get('tipping_marker','?')}, "
        f"U:Cr={uc_ratio} ({uc_mechanism}), "
        f"IL-10 Energy={il10_energy:.4f} (steroids {steroid_status})"
    )

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":    CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Claude API error {response.status_code}: {response.text[:300]}"
        )

    synthesis_text = response.json()["content"][0]["text"].strip()
    synthesis_text += INVESTIGATIONAL_DISCLAIMER

    return {
        "synthesis_text":     synthesis_text,
        "hard_rules_applied": hard_rules_applied,
        "model_used":         CLAUDE_MODEL,
        "rag_chunk_count":    len(chunks),
        "uc_ratio":           uc_ratio,
        "uc_mechanism":       uc_mechanism,
        "il10_energy":        il10_energy,
        "steroid_status":     steroid_status,
    }
