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


# ── Biological system groups ──────────────────────────────────────────────────

SYSTEM_GROUPS: Dict[str, str] = {
    "D-dimer":    "COAGULATION",
    "Platelets":  "COAGULATION",
    "CRP":        "INFLAMMATION",
    "IL-6":       "INFLAMMATION",
    "IL-8":       "INFLAMMATION",
    "IL-10":      "INFLAMMATION",
    "PTX3":       "INFLAMMATION",
    "TNF-alpha":  "INFLAMMATION",
    "IL-18":      "INFLAMMATION",
    "sIL-1R2":    "INFLAMMATION",
    "IL-1ra":     "INFLAMMATION",
    "Creatinine": "ORGAN FUNCTION",
    "Urea":       "ORGAN FUNCTION",
    "SOFA":       "ORGAN FUNCTION",
    "MAP":        "HAEMODYNAMICS",
    "Lactate":    "HAEMODYNAMICS",
    "PaO2_FiO2":  "OXYGENATION",
    "GCS":        "NEUROLOGICAL",
}

_SYSTEM_ORDER = [
    "HAEMODYNAMICS", "COAGULATION", "ORGAN FUNCTION",
    "INFLAMMATION", "OXYGENATION", "NEUROLOGICAL",
]

# ── Clinical reference ranges ─────────────────────────────────────────────────
# (lower_normal, upper_normal, unit, direction)
# direction: "high_bad" = elevated is pathological; "low_bad" = low is pathological

_CLINICAL_REFS: Dict[str, tuple] = {
    "Lactate":    (0.5,   2.0,   "mmol/L",  "high_bad"),
    "SOFA":       (0,     2,     "points",  "high_bad"),
    "PTX3":       (0,     2.0,   "ng/mL",   "high_bad"),
    "D-dimer":    (0,     500,   "ng/mL",   "high_bad"),
    "IL-10":      (0,     9.1,   "pg/mL",   "high_bad"),
    "Creatinine": (44,    110,   "μmol/L",  "high_bad"),
    "Urea":       (2.5,   7.1,   "mmol/L",  "high_bad"),
    "Platelets":  (150,   400,   "×10⁹/L",  "low_bad"),
    "MAP":        (70,    105,   "mmHg",    "low_bad"),
    "GCS":        (15,    15,    "points",  "low_bad"),
    "CRP":        (0,     5,     "mg/L",    "high_bad"),
    "PaO2_FiO2":  (300,   500,   "ratio",   "low_bad"),
    "IL-8":       (0,     16,    "pg/mL",   "high_bad"),
    "IL-6":       (0,     7,     "pg/mL",   "high_bad"),
    "sIL-1R2":    (0,     2000,  "pg/mL",   "high_bad"),
    "IL-18":      (0,     200,   "pg/mL",   "high_bad"),
    "IL-1ra":     (0,     200,   "pg/mL",   "high_bad"),
    "TNF-alpha":  (0,     8.1,   "pg/mL",   "high_bad"),
}


def _ref_deviation_label(marker: str, raw_val) -> str:
    """Return a short label showing deviation from clinical reference range."""
    if raw_val is None:
        return ""
    ref = _CLINICAL_REFS.get(marker)
    if ref is None:
        return ""
    lo, hi, unit, direction = ref
    val = float(raw_val)
    if direction == "high_bad":
        if val > hi and hi > 0:
            fold = val / hi
            if fold >= 2.0:
                return f"{fold:.1f}× ULN"
            pct = (val - hi) / hi * 100
            return f"+{pct:.0f}% above ULN"
        elif val < lo:
            return f"below ref ({lo} {unit})"
        return "within ref"
    else:  # low_bad
        if val < lo and lo > 0:
            pct = (lo - val) / lo * 100
            return f"{pct:.0f}% below LLN"
        return "within ref" if val <= hi else f"above ref ({hi} {unit})"


# ── Hard clinical rules: PASS / TRIGGERED / INDETERMINATE ─────────────────────

def evaluate_hard_rules(raw: Dict[str, Any], ep: Dict[str, float]) -> List[dict]:
    """
    Evaluate the 5 hard clinical rules from patient biomarker values.
    Mirrors the logic in treatment.py — deterministic, no API call.

    Returns list of dicts: {rule_id, name, status, detail}
    status ∈ {"PASS", "TRIGGERED", "INDETERMINATE"}
    """
    rules: List[dict] = []

    # ── Rule 1 + 5: U:Cr ratio → catabolism vs AKI + protein target ─────────
    urea      = float(raw.get("Urea", 0) or 0)
    creat     = float(raw.get("Creatinine", 1) or 1)
    creat_si  = creat / 1000.0          # μmol/L → mmol/L
    uc_ratio  = round(urea / creat_si, 1) if creat_si > 0 else 0.0
    catabolic = uc_ratio > 150

    rules.append({
        "rule_id": "R1",
        "name":    "U:Cr Catabolism (not AKI)",
        "status":  "TRIGGERED" if catabolic else "PASS",
        "detail":  (
            f"U:Cr={uc_ratio} >150 → CATABOLIC HYPERUREAEMIA. "
            "Lead with proteolysis, not renal failure. "
            "Do NOT frame as AKI-first."
            if catabolic else
            f"U:Cr={uc_ratio} — AKI-predominant pattern. Renal-first approach appropriate."
        ),
    })

    # ── Rule 2: IL-10 energy → steroid contraindication ──────────────────────
    il10_e  = ep.get("IL-10", 0.0)
    steroids_ci = il10_e > 0.70
    rules.append({
        "rule_id": "R2",
        "name":    "IL-10 / Steroid Contraindication",
        "status":  "TRIGGERED" if steroids_ci else "PASS",
        "detail":  (
            f"IL-10 energy={il10_e:.4f} >0.70 → Corticosteroids CONTRAINDICATED. "
            "Anti-inflammatory reserve intact; steroids increase secondary infection risk."
            if steroids_ci else
            f"IL-10 energy={il10_e:.4f} — Steroid candidacy: individual haemodynamic assessment required."
        ),
    })

    # ── Rule 3: DIC — ISTH score required (D-dimer alone is insufficient) ────
    ddimer = float(raw.get("D-dimer", 0) or 0)
    plt    = float(raw.get("Platelets", 999) or 999)
    if ddimer > 3000 and plt < 100:
        r3_status = "TRIGGERED"
        r3_detail = (
            f"D-dimer={ddimer:.0f} ng/mL + Platelets={plt:.0f} ×10⁹/L — "
            "Full ISTH DIC score ≥5 required. D-dimer alone does NOT establish DIC."
        )
    elif ddimer > 500:
        r3_status = "INDETERMINATE"
        r3_detail = (
            f"D-dimer={ddimer:.0f} ng/mL (above 500 threshold) — "
            "Obtain ISTH DIC score before diagnosing DIC."
        )
    else:
        r3_status = "PASS"
        r3_detail = f"D-dimer={ddimer:.0f} ng/mL — DIC threshold not reached on current data."
    rules.append({
        "rule_id": "R3",
        "name":    "DIC — ISTH Score Required",
        "status":  r3_status,
        "detail":  r3_detail,
    })

    # ── Rule 4: CRRT indication per KDIGO (Urea alone is NOT sufficient) ─────
    rules.append({
        "rule_id": "R4",
        "name":    "CRRT per KDIGO (not Urea target)",
        "status":  "PASS",
        "detail":  (
            f"Urea={urea} mmol/L — CRRT requires KDIGO criteria: "
            "refractory fluid overload, pH <7.15, K⁺ >6.5, uraemic complications, "
            "or vasopressor-refractory instability. "
            "Urea level alone is NOT an indication."
        ),
    })

    # ── Rule 5: Protein target (driven by Rule 1 result) ─────────────────────
    rules.append({
        "rule_id": "R5",
        "name":    "Protein Target (catabolic adjustment)",
        "status":  "TRIGGERED" if catabolic else "PASS",
        "detail":  (
            f"U:Cr={uc_ratio} (catabolic) → Protein ≥1.5 g/kg/day (ESPEN 2019, Heyland 2011). "
            "Do NOT restrict to 0.8–1.2 g/kg/day — protein restriction in catabolic sepsis "
            "accelerates muscle wasting and INCREASES mortality."
            if catabolic else
            f"U:Cr={uc_ratio} — Standard protein 1.2–1.5 g/kg/day. Monitor renal tolerance."
        ),
    })

    return rules


# ── Phenotype classification: SRS1 (immunoparalysis) vs SRS2 (hyperinflammatory) ─

def classify_phenotype(raw: Dict[str, Any], ep: Dict[str, float]) -> dict:
    """
    Classify sepsis immune phenotype from IL-10 energy, CRP, SOFA.

    SRS2-like  = Hyperinflammatory:  high CRP, IL-10 not dominant, high SOFA
    SRS1-like  = Immunoparalysis:    IL-10 energy dominant (>0.70), high SOFA
    Mixed      = Both arms active simultaneously
    Indeterminate = SOFA <6 (insufficient severity for confident classification)

    Returns dict: {phenotype, subtype, confidence, evidence, clinical_implications}
    """
    il10_e   = ep.get("IL-10", 0.0)
    il10_raw = float(raw.get("IL-10", 0) or 0)    # pg/mL — use RAW for phenotype
    crp_val  = float(raw.get("CRP", 0) or 0)
    sofa_val = float(raw.get("SOFA", 0) or 0)

    # NOTE: phenotype classification uses RAW biomarker values (clinical thresholds).
    # Rule R2 steroid contraindication uses IL-10 ENERGY — these are separate questions:
    #   R2: "Is anti-inflammatory reserve working?" → IL-10 energy
    #   Phenotype: "Which immune state?" → raw IL-10 concentration vs CRP

    if sofa_val < 6:
        return {
            "phenotype":            "INDETERMINATE",
            "subtype":              "—",
            "confidence":           "LOW (SOFA <6)",
            "evidence":             [f"SOFA={sofa_val:.0f} — insufficient severity for SRS phenotyping"],
            "clinical_implications": ["Phenotype classification requires SOFA ≥6"],
        }

    # IMMUNOPARALYSIS: markedly elevated raw IL-10 (>50 pg/mL) — immune exhaustion signal
    if il10_raw > 50 and sofa_val >= 6:
        phenotype   = "IMMUNOPARALYSIS"
        subtype     = "SRS1-like"
        confidence  = "HIGH" if (il10_raw > 100 and crp_val < 100) else "MODERATE"
        evidence    = [
            f"IL-10={il10_raw:.1f} pg/mL (>50) — markedly elevated: immune exhaustion likely",
            f"CRP={crp_val:.0f} mg/L (may be suppressed by IL-10 override)",
            f"SOFA={sofa_val:.0f} — organ failure without proportionate inflammatory signal",
        ]
        implications = [
            "⚠ Corticosteroids CONTRAINDICATED (Rule R2) — deepens immunosuppression",
            "Risk: secondary bacterial/fungal infection, failure to clear primary source",
            "Consider: IFN-γ or GM-CSF protocol if immunoparalysis confirmed on serial testing",
        ]

    # HYPERINFLAMMATORY: high CRP, IL-10 not compensating, severe SOFA
    elif crp_val > 150 and il10_raw < 50 and sofa_val >= 6:
        phenotype   = "HYPERINFLAMMATORY"
        subtype     = "SRS2-like"
        confidence  = "HIGH" if crp_val > 200 else "MODERATE"
        evidence    = [
            f"CRP={crp_val:.0f} mg/L (>150) — sustained high-grade acute-phase response",
            f"IL-10={il10_raw:.1f} pg/mL (<50) — compensatory immunosuppression NOT dominant",
            f"SOFA={sofa_val:.0f} — multi-organ dysfunction confirms systemic severity",
        ]
        implications = [
            "Steroid candidacy: individual haemodynamic assessment (not automatically indicated)",
            "Risk: cytokine storm, immunothrombosis, DIC propagation",
            "Consider: tocilizumab / anakinra if inflammation escalating despite source control",
        ]

    elif sofa_val >= 6:
        # Both CRP elevated AND IL-10 rising — transitional state
        phenotype   = "MIXED / TRANSITIONAL"
        subtype     = "SRS1–SRS2 overlap"
        confidence  = "LOW"
        evidence    = [
            f"CRP={crp_val:.0f} mg/L + IL-10={il10_raw:.1f} pg/mL — "
            "neither arm clearly dominant",
            f"SOFA={sofa_val:.0f} — severity confirmed",
            "Pattern: hyperinflammation with emerging compensatory immunosuppression",
        ]
        implications = [
            "Serial IL-10 tracking critical — direction of drift determines therapy",
            "Steroid candidacy: INDETERMINATE — monitor 12–24h trend before decision",
            "Phenotype likely to declare within 24h — repeat BEI at next labs",
        ]
    else:
        phenotype   = "INDETERMINATE"
        subtype     = "—"
        confidence  = "LOW"
        evidence    = [f"SOFA={sofa_val:.0f}, CRP={crp_val:.0f}, IL-10={il10_raw:.1f} — pattern unclear"]
        implications = ["Insufficient data for confident phenotype assignment"]

    return {
        "phenotype":            phenotype,
        "subtype":              subtype,
        "confidence":           confidence,
        "evidence":             evidence,
        "clinical_implications": implications,
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
            "system_group":    SYSTEM_GROUPS.get(marker, "OTHER"),
            "ref_deviation":   _ref_deviation_label(marker, raw_val),
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

    # ── Hard rules + phenotype (deterministic enrichment) ────────────────────
    hard_rules = evaluate_hard_rules(raw, ep)
    phenotype  = classify_phenotype(raw, ep)

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
        "hard_rules":         hard_rules,
        "phenotype":          phenotype,
    }


def format_xai_for_prompt(xai: dict, max_markers: int = 17) -> str:
    """
    Clinical explainability block for display or prompt injection.
    Sections:
      1. Phenotype flag (SRS1/SRS2/Mixed)
      2. Hard rules PASS/TRIGGERED/INDETERMINATE header
      3. Per-marker attribution table grouped by biological system
         — includes reference range deviation magnitude
    """
    lines = [
        "═══ SepsisAI xAI EXPLAINABILITY REPORT ════════════════════════════════════",
        f"  BEI Score : {xai['bei_score']:.4f}   Zone: {xai['bei_zone']}   Module {xai['module']}",
        f"  Summary   : {xai['xai_summary']}",
        "",
    ]

    # ── Section 1: Phenotype flag ─────────────────────────────────────────────
    ph = xai.get("phenotype", {})
    if ph:
        pstatus = ph.get("phenotype", "UNKNOWN")
        psub    = ph.get("subtype", "")
        pconf   = ph.get("confidence", "")
        lines += [
            "┌─ IMMUNE PHENOTYPE ─────────────────────────────────────────────────────────",
            f"│  {pstatus}  ({psub})   Confidence: {pconf}",
        ]
        for ev in ph.get("evidence", []):
            lines.append(f"│  Evidence  : {ev}")
        for imp in ph.get("clinical_implications", []):
            lines.append(f"│  ► {imp}")
        lines.append("└───────────────────────────────────────────────────────────────────────────")
        lines.append("")

    # ── Section 2: Hard rules header ─────────────────────────────────────────
    hard_rules = xai.get("hard_rules", [])
    if hard_rules:
        lines.append("┌─ HARD CLINICAL RULES ──────────────────────────────────────────────────────")
        for r in hard_rules:
            status_icon = "⚠ TRIGGERED" if r["status"] == "TRIGGERED" else (
                "? INDET.   " if r["status"] == "INDETERMINATE" else "✓ PASS     "
            )
            lines.append(f"│  [{r['rule_id']}] {status_icon}  {r['name']}")
            lines.append(f"│           {r['detail']}")
        lines.append("└───────────────────────────────────────────────────────────────────────────")
        lines.append("")

    # ── Section 3: Attribution table, grouped by biological system ────────────
    lines.append(f"┌─ PER-MARKER ATTRIBUTION  (Module {xai['module']} — worst drivers first) ──────────")
    col_header = (
        f"│  {'Marker':<14} {'TC':>9}  {'Energy':>7}  {'Severity':<12}  "
        f"{'System':<15}  {'Raw Value':<12}  Ref Deviation"
    )
    lines.append(col_header)
    lines.append("│  " + "─" * 90)

    # Group markers by system, preserving TC sort within each group
    attributions = xai["marker_attributions"][:max_markers]

    # Build system → markers map (sorted by TC within each system)
    from collections import defaultdict
    sys_map: dict = defaultdict(list)
    for a in attributions:
        sys_map[a.get("system_group", "OTHER")].append(a)

    # Print in canonical system order, remaining systems after
    printed_systems: set = set()
    for sys in _SYSTEM_ORDER:
        group = sys_map.get(sys, [])
        if not group:
            continue
        printed_systems.add(sys)
        lines.append(f"│  ── {sys} {'─' * (85 - len(sys))}")
        for a in group:
            primary_flag  = " ◄ PRIMARY" if a["is_primary"] else ""
            raw_display   = str(a["raw_value"]) if a["raw_value"] is not None else "MISSING"
            ref_dev       = a.get("ref_deviation", "")
            lines.append(
                f"│  {a['marker']:<14} {a['tc']:>+9.5f}  {a['energy']:>7.4f}  "
                f"{a['severity']:<12}  {sys:<15}  {raw_display:<12}  {ref_dev}{primary_flag}"
            )

    # Any markers not in _SYSTEM_ORDER
    for sys, group in sys_map.items():
        if sys not in printed_systems:
            lines.append(f"│  ── {sys} {'─' * (85 - len(sys))}")
            for a in group:
                primary_flag = " ◄ PRIMARY" if a["is_primary"] else ""
                raw_display  = str(a["raw_value"]) if a["raw_value"] is not None else "MISSING"
                ref_dev      = a.get("ref_deviation", "")
                lines.append(
                    f"│  {a['marker']:<14} {a['tc']:>+9.5f}  {a['energy']:>7.4f}  "
                    f"{a['severity']:<12}  {sys:<15}  {raw_display:<12}  {ref_dev}{primary_flag}"
                )

    lines.append("│")
    lines.append(f"│  OUTCOME: {xai['predicted_outcome']}")
    lines.append("└───────────────────────────────────────────────────────────────────────────")
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
