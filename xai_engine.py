"""
xai_engine.py — Explainability (xAI) Layer for SepsisAI BEI
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Provides:
  1. build_xai_attribution(bei, raw)
       Pure-Python structured attribution block.
       Per-marker TC breakdown with clinical language, tier context,
       and deviation from cohort — no API call required.

  2. format_xai_for_prompt(xai)
       Compact string representation for injection into treatment/rounds prompts.

  3. build_rounds_prompt(bei, raw, xai)
       Doctor-on-rounds Claude API prompt.
       Asks Claude to think as an attending reviewing the morning labs:
       WHY this prediction, WHAT would you say at the bedside, WHAT next.

  4. synthesize_rounds_report(data, api_key)
       Calls Claude API with the rounds prompt.
       Returns structured dict with report text + metadata.

Design note: xAI attribution is deliberately decoupled from the Claude API call.
The attribution block is deterministic (pure math from bei_engine.py output) and
can be serialised independently. The rounds narrative is Claude-generated but
grounded in the deterministic attribution, not the other way around.

Davoudian et al. 2023 (PLOS ONE, N=135, AUC=0.812) is the evidence anchor.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import httpx

from bei_engine import (
    BEI_TIPPING_TIERS,
    BEI_COHORT_ENERGY_MEANS,
    BEI_WEIGHTS,
    BEI_FAILURE_MARKERS,
    BEI_PROTECTIVE_MARKERS,
    BEI_PERFORMANCE,
    BEI_ZONES,
    KILL4_WEIGHTS,
)
from config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, INVESTIGATIONAL_DISCLAIMER

logger = logging.getLogger(__name__)


# ── Clinical language templates ───────────────────────────────────────────────

_MARKER_CLINICAL_CONTEXT: Dict[str, str] = {
    "D-dimer":    "Coagulation cascade activation marker. Elevated D-dimer signals systemic fibrinolysis and endothelial microthrombi formation.",
    "SOFA":       "Composite multi-organ dysfunction score. Each point reflects progressive failure across hepatic, renal, respiratory, neurological, coagulation, and cardiovascular systems.",
    "Lactate":    "Tissue oxygen debt marker. Elevated lactate is direct evidence of anaerobic metabolism — cells are failing to produce ATP aerobically.",
    "PTX3":       "Endothelial-innate immunity pentraxin. Elevated PTX3 marks activated endothelium with neutrophil-driven vascular injury; independent of CRP.",
    "Urea":       "Nitrogen waste marker. Elevated urea in sepsis more often reflects catabolic proteolysis (muscle breakdown) than primary renal failure; U:Cr ratio distinguishes the two.",
    "IL-8":       "Neutrophil chemotactic cytokine. Drives acute neutrophilic inflammation; indicator of active bacterial/fungal infection burden and tissue neutrophilia.",
    "IL-10":      "Anti-inflammatory regulatory cytokine. Elevated IL-10 signals compensatory immunosuppression — the immune system is throttling down to avoid collateral damage.",
    "GCS":        "Neurological consciousness score. GCS ≤14 indicates cerebral hypoperfusion or encephalopathy; below this threshold, 90-day mortality rises sharply.",
    "sIL-1R2":    "Soluble IL-1 receptor type 2 — decoy receptor that traps IL-1β. Elevated levels signal late-phase NLRP3 inflammasome activity.",
    "CRP":        "Acute-phase reactant (liver-derived). Broad inflammation marker; lags 12–24h; useful for trend monitoring rather than acute triage.",
    "PaO2_FiO2":  "Oxygenation ratio (Horowitz Index). Values <300 define ALI; <200 define ARDS. Directly reflects lung parenchymal reserve.",
    "IL-6":       "Pleiotropic cytokine; elevated across virtually all sepsis phenotypes. Downstream effector — present but does not independently predict tipping.",
    "MAP":        "Mean arterial pressure — haemodynamic adequacy proxy. MAP <65 mmHg triggers vasopressor protocol; <55 mmHg signals shock.",
    "Platelets":  "Thrombocyte count. Thrombocytopaenia (<100×10⁹/L) in sepsis indicates DIC, bone marrow suppression, or platelet consumption.",
    "IL-18":      "Inflammasome-derived cytokine (NLRP3 pathway). Informationally correlated with sIL-1R2; sIL-1R2 is the preferred NLRP3 marker.",
    "IL-1ra":     "IL-1 receptor antagonist — endogenous anti-cytokine. Elevated IL-1ra reflects the immune system mounting a futile counter-regulatory response.",
    "TNF-alpha":  "Macrophage-derived pro-inflammatory cytokine. Amplifies inflammation through NF-κB; elevated levels suggest ongoing macrophage activation.",
    "Creatinine": "Renal filtration marker. Elevated creatinine indicates GFR reduction; interpret alongside urea (U:Cr ratio) to determine whether failure is intrinsic or pre-renal.",
}


def build_xai_attribution(bei: dict, raw: Dict[str, Any]) -> dict:
    """
    Build structured per-marker xAI attribution from a BEI result.

    Pure computation — no API call. Deterministic given the same BEI output.

    Returns
    -------
    dict with keys:
      module              : "A" or "B"
      bei_score           : float
      bei_zone            : str
      predicted_outcome   : str  — natural language prediction statement
      zone_confidence     : str  — calibration note
      top_driver          : dict — primary failure marker with full attribution
      marker_attributions : list — all markers sorted worst TC first
      xai_summary         : str  — 2-sentence plain-English explanation
    """
    module      = bei.get("module", "A")
    score       = bei.get("bei_score", 0.0)
    zone        = bei.get("bei_zone", "UNKNOWN")
    ep          = bei.get("energy_profile", {})
    tc          = bei.get("tipping_contributions", {})
    tipping_m   = bei.get("tipping_marker", "")

    # Determine which weights were used
    weights = KILL4_WEIGHTS if module == "B" else BEI_WEIGHTS

    # ── Per-marker attribution ────────────────────────────────────────────────
    attributions: List[dict] = []
    for marker, weight in sorted(weights.items(), key=lambda x: -abs(x[1])):
        energy      = ep.get(marker)
        if energy is None:
            continue
        cohort_mean = BEI_COHORT_ENERGY_MEANS.get(marker, 0.60)
        tc_val      = tc.get(marker, 0.0)
        tier_info   = BEI_TIPPING_TIERS.get(marker, ("UNKNOWN", "—"))
        tier        = tier_info[0]
        is_failure  = marker in BEI_FAILURE_MARKERS
        direction   = "FAILURE" if is_failure else "PROTECTIVE"

        # Clinical deviation label
        deviation   = energy - cohort_mean
        if abs(deviation) < 0.05:
            deviation_label = "near cohort mean"
        elif deviation < 0:
            dev_pct = abs(deviation) / max(cohort_mean, 0.01) * 100
            deviation_label = f"{dev_pct:.0f}% below cohort mean — WORSE than average"
        else:
            dev_pct = deviation / max(cohort_mean, 0.01) * 100
            deviation_label = f"{dev_pct:.0f}% above cohort mean — better than average"

        # Severity label
        if tc_val < -0.03:
            severity = "CRITICAL"
        elif tc_val < -0.01:
            severity = "SIGNIFICANT"
        elif tc_val < 0:
            severity = "MILD"
        elif tc_val >= 0.01:
            severity = "PROTECTIVE"
        else:
            severity = "NEUTRAL"

        raw_val = raw.get(marker)
        raw_str = f"{raw_val}" if raw_val is not None else "NOT PROVIDED"

        attributions.append({
            "marker":          marker,
            "raw_value":       raw_val,
            "raw_str":         raw_str,
            "energy":          round(energy, 4),
            "cohort_mean":     round(cohort_mean, 4),
            "deviation":       round(deviation, 4),
            "deviation_label": deviation_label,
            "weight":          round(weight, 6),
            "tc":              round(tc_val, 6),
            "tier":            tier,
            "direction":       direction,
            "severity":        severity,
            "is_primary":      marker == tipping_m,
            "clinical_context": _MARKER_CLINICAL_CONTEXT.get(marker, ""),
        })

    # Sort: most negative TC first (worst drivers at top)
    attributions.sort(key=lambda x: x["tc"])

    # ── Outcome prediction statement ─────────────────────────────────────────
    zone_meta   = BEI_ZONES.get(zone, (0, 1, "Unknown zone"))
    zone_detail = zone_meta[2] if isinstance(zone_meta, tuple) and len(zone_meta) > 2 else ""

    # Quartile lookup (BEI_PERFORMANCE has Q1–Q5 mortality rates)
    # Q1 = lowest BEI (≤25th pct), Q5 = highest BEI (>80th pct)
    # Approximate quartile from score
    bp = BEI_PERFORMANCE
    if score <= 0.45:
        quartile_mortality = bp.get("Q1_mortality", 0.90)
        quartile_label     = "1st quartile (lowest biological reserve)"
    elif score <= 0.55:
        quartile_mortality = bp.get("Q2_mortality", 0.43)
        quartile_label     = "2nd quartile"
    elif score <= 0.67:
        quartile_mortality = 0.15   # between Q2 and Q3/Q4 (AT_RISK zone)
        quartile_label     = "3rd quartile"
    elif score <= 0.80:
        quartile_mortality = bp.get("Q3_Q4_mortality", 0.10)
        quartile_label     = "3rd–4th quartile"
    else:
        quartile_mortality = bp.get("Q5_mortality", 0.19)
        quartile_label     = "5th quartile (highest biological reserve)"

    predicted_outcome = (
        f"BEI={score:.4f} ({zone} zone) places this patient in the {quartile_label} "
        f"of the Davoudian 2023 cohort (N=135). "
        f"Patients in this quartile had approximately {quartile_mortality*100:.0f}% 90-day mortality. "
        f"Cohort baseline: {bp['cohort_baseline']*100:.1f}%. "
        f"{'This BEI places the patient BELOW the cohort baseline — elevated risk.' if quartile_mortality > bp['cohort_baseline'] else 'This BEI places the patient ABOVE the cohort baseline — relative preservation of energy reserve.'}"
    )

    zone_confidence = (
        f"Module {module} AUC={bp['auc_entropy_weighted'] if module == 'A' else 'not independently validated — Module A AUC={:.3f} as upper bound'.format(bp['auc_entropy_weighted'])}. "
        f"Zone thresholds: VERY_LOW <0.55, AT_RISK 0.55–0.67, SUFFICIENT >0.67. "
        f"Calibration cohort: {bp['cohort']}."
    )

    # ── Top driver block ─────────────────────────────────────────────────────
    top_driver = next((a for a in attributions if a["is_primary"]), attributions[0] if attributions else {})

    # ── Plain-English xAI summary (2 sentences) ──────────────────────────────
    if top_driver:
        tm   = top_driver["marker"]
        ttc  = top_driver["tc"]
        tdev = top_driver["deviation_label"]
        xai_summary = (
            f"The model predicts {zone} zone because {tm} (TC={ttc:+.4f}) is the primary "
            f"failure driver — {tdev}, pulling biological energy below the survival threshold. "
            f"The prediction is grounded in entropy-weighted attribution across "
            f"{len(attributions)} {'(full 17-marker)' if module == 'A' else '(KILL-4)'} "
            f"biomarkers calibrated against {bp['cohort']}."
        )
    else:
        xai_summary = f"BEI={score:.4f} ({zone} zone). Insufficient marker data for attribution."

    return {
        "module":             module,
        "bei_score":          score,
        "bei_zone":           zone,
        "predicted_outcome":  predicted_outcome,
        "zone_confidence":    zone_confidence,
        "top_driver":         top_driver,
        "marker_attributions": attributions,
        "xai_summary":        xai_summary,
        "n_markers":          len(attributions),
    }


def format_xai_for_prompt(xai: dict, max_markers: int = 8) -> str:
    """
    Compact string representation for injection into a Claude prompt.
    Shows the xAI summary, prediction, and top N markers sorted by TC.
    """
    lines = [
        "═══ xAI ATTRIBUTION BLOCK ══════════════════════════════════",
        f"BEI Score   : {xai['bei_score']:.4f}  Zone: {xai['bei_zone']}",
        f"xAI Summary : {xai['xai_summary']}",
        "",
        f"PREDICTED OUTCOME: {xai['predicted_outcome']}",
        "",
        f"PER-MARKER ATTRIBUTION (worst drivers first, Module {xai['module']}):",
        f"{'Marker':<14} {'TC':>9}  {'Energy':>7}  {'CohtMean':>8}  {'Severity':<12}  {'Tier':<10}  Raw Value",
        "─" * 90,
    ]

    for a in xai["marker_attributions"][:max_markers]:
        marker_flag = "← PRIMARY" if a["is_primary"] else ""
        raw_display = str(a["raw_value"]) if a["raw_value"] is not None else "MISSING"
        lines.append(
            f"{a['marker']:<14} {a['tc']:>+9.5f}  {a['energy']:>7.4f}  {a['cohort_mean']:>8.4f}  "
            f"{a['severity']:<12}  {a['tier']:<10}  {raw_display}  {marker_flag}"
        )

    lines.append("═" * 70)
    return "\n".join(lines)


def build_rounds_prompt(
    bei:    dict,
    raw:    Dict[str, Any],
    xai:    dict,
    pid:    str = "UNKNOWN",
    flags:  Optional[List[str]] = None,
) -> str:
    """
    Doctor-on-rounds Claude prompt.

    Asks Claude to reason as an ICU attending reviewing morning labs:
    1. Why is the model predicting this? (ground in xAI attribution)
    2. What is the biological story connecting these markers?
    3. What would you say to the team on rounds?
    4. What are the 3 most important next clinical actions?

    This is separate from treatment.py (which generates protocol-level treatment).
    The rounds report is narrative — the attending's verbal reasoning + immediate priorities.
    """
    xai_block     = format_xai_for_prompt(xai)
    zone          = bei.get("bei_zone", "?")
    score         = bei.get("bei_score", 0.0)
    tipping       = bei.get("tipping_marker", "?")
    tipping_tier  = bei.get("tipping_tier", "?")
    zone_detail   = bei.get("bei_zone_detail", "")
    flags_str     = "\n".join(f"  ⚠ {f}" for f in (flags or [])) or "  None"
    top_driver    = xai.get("top_driver", {})
    prediction    = xai.get("predicted_outcome", "")

    # Build the 3 worst markers + their clinical context for the prompt
    worst_3 = xai["marker_attributions"][:3]
    worst_str = "\n".join(
        f"  {a['marker']}: TC={a['tc']:+.5f}, Energy={a['energy']:.4f}, "
        f"Raw={a['raw_str']} — {a['clinical_context']}"
        for a in worst_3
    )

    return f"""You are SepsisAI, a clinical decision-support AI built by Adnexus Biotechnologies Inc.
You are helping an ICU attending physician prepare for morning rounds.
This is INVESTIGATIONAL — not FDA cleared.

Your role: Think as the attending. Look at these labs, look at the model's explanation,
and produce the kind of clear, grounded verbal briefing you would give to your team
at the bedside — not a protocol document, not a template, not a list of generic
sepsis interventions. This patient, these numbers, this biological story.

══════════════════════════════════════════════════════════════
PATIENT: {pid}
══════════════════════════════════════════════════════════════

{xai_block}

TOP 3 FAILING DIMENSIONS — CLINICAL CONTEXT:
{worst_str}

ACTIVE CLINICAL FLAGS:
{flags_str}

ZONE DETAIL: {zone_detail}

OUTCOME PREDICTION:
{prediction}

PRIMARY TIPPING MARKER: {tipping} ({tipping_tier}-tier)
  The model identifies {tipping} as the marker pulling this patient's biological
  energy most below the cohort average. This is the rate-limiting failure dimension.
══════════════════════════════════════════════════════════════

YOUR TASK — produce EXACTLY these four sections. No preamble. Start with ROUNDS BRIEFING.

ROUNDS BRIEFING (2–3 sentences)
Write the verbal opening you would say to your team standing at this patient's bedside.
Ground it in the BEI zone, the tipping marker, and what the trajectory means clinically.
Sound like an attending, not a report. "This patient's biology is telling us..." is the frame.

WHY THIS PREDICTION
In 3–5 sentences, explain why the model scores this patient as {zone} zone.
Walk through the xAI attribution: what is each of the top failing markers doing
physiologically, and how do they connect? This is the "model transparency" section —
a clinician reading this should understand not just what the score is, but why.
Use the TC values as evidence: TC={top_driver.get('tc', 0):+.5f} for {tipping} means [explain what
that means in terms of deviation from the average sepsis patient].

BIOLOGICAL NARRATIVE
One paragraph. Connect the top failing markers into a coherent physiological story.
Is this DIC driving multi-organ failure? Immunoparalysis preventing clearance?
Endothelial collapse from PTX3 + D-dimer? Catabolic hyperureaemia masking renal injury?
Name the mechanism. The team needs to understand the pathophysiology, not just the numbers.

IMMEDIATE PRIORITIES — 3 ACTIONS
What are the 3 most important things this team needs to do in the next 6 hours?
Be specific. Not "obtain blood cultures" — that is already done. Target the biology
the model flagged. Frame each as: Action | Biological rationale | Target endpoint.

{INVESTIGATIONAL_DISCLAIMER}"""


def synthesize_rounds_report(
    data:    dict,
    api_key: str,
) -> dict:
    """
    Call Claude API with the doctor-on-rounds prompt.

    Parameters
    ----------
    data : dict with keys:
        bei        : BEI result dict from compute_bei() / compute_bei_b()
        raw_input  : raw marker dict
        patient_id : str
        flags      : List[str] (clinical flags from build_clinical_flags)

    Returns
    -------
    dict:
        rounds_text     : str   — full narrative report
        xai_attribution : dict  — structured attribution (deterministic)
        model_used      : str
        bei_zone        : str
        bei_score       : float
        tipping_marker  : str
    """
    bei    = data["bei"]
    raw    = data["raw_input"]
    pid    = data.get("patient_id", "UNKNOWN")
    flags  = data.get("flags", [])

    # Build xAI attribution first (deterministic — no API)
    xai    = build_xai_attribution(bei, raw)
    prompt = build_rounds_prompt(bei, raw, xai, pid=pid, flags=flags)

    logger.info(
        f"Rounds report: patient={pid}, zone={bei.get('bei_zone','?')}, "
        f"tipping={bei.get('tipping_marker','?')}, "
        f"n_markers={xai['n_markers']}"
    )

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model":      CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120.0,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Claude API error {response.status_code}: {response.text[:300]}"
        )

    rounds_text = response.json()["content"][0]["text"].strip()

    return {
        "rounds_text":     rounds_text,
        "xai_attribution": xai,
        "model_used":      CLAUDE_MODEL,
        "bei_zone":        bei.get("bei_zone", "UNKNOWN"),
        "bei_score":       bei.get("bei_score", 0.0),
        "tipping_marker":  bei.get("tipping_marker", ""),
    }
