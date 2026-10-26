"""
bei_engine.py — Biological Energy Index computation engine
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Based on: Davoudian et al. 2023 (PLOS ONE, N=135, AUC=0.812)
BEI formula: Σ[W(m) × Energy_m(i)] / Σ[W(m)]
             where Energy is entropy-weighted normalised marker score [0,1]

Module A: Full 17-marker BEI  (AUC=0.812, N=135, Davoudian 2023)
Module B: KILL-4 reduced panel (D-dimer, SOFA, Lactate, PTX3)
          AUC not independently validated — use Module A AUC as upper bound.
          Zone thresholds derived from Module A calibration; validate before deployment.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

# ── Module A — entropy-derived weights (Davoudian 2023 approximate) ───────────
BEI_WEIGHTS_RAW: Dict[str, float] = {
    "D-dimer":   0.113, "SOFA":      0.099, "Lactate":   0.095,
    "PTX3":      0.089, "Urea":      0.080, "IL-8":      0.072,
    "IL-10":     0.069, "GCS":       0.056, "sIL-1R2":   0.053,
    "CRP":       0.041, "PaO2_FiO2": 0.039, "IL-6":      0.036,
    "MAP":       0.034, "Platelets": 0.031, "IL-18":     0.029,
    "IL-1ra":    0.026, "TNF-alpha": 0.023, "Creatinine": 0.015,
}
_w_total = sum(BEI_WEIGHTS_RAW.values())
BEI_WEIGHTS: Dict[str, float] = {
    k: round(v / _w_total, 6) for k, v in BEI_WEIGHTS_RAW.items()
}

# Markers where HIGH value = failure (energy is INVERTED after normalisation)
BEI_FAILURE_MARKERS = frozenset({
    "D-dimer", "SOFA", "Lactate", "PTX3", "Urea", "IL-8", "IL-10",
    "IL-6", "IL-1ra", "TNF-alpha", "IL-18", "CRP", "sIL-1R2", "Creatinine",
})
# Markers where HIGH value = protective (energy NOT inverted)
BEI_PROTECTIVE_MARKERS = frozenset({"GCS", "MAP", "PaO2_FiO2", "Platelets"})

# Calibrated log10 range [lo, hi] from Davoudian 2023 cohort (N=135)
BEI_COHORT_LOG_RANGE: Dict[str, Tuple[float, float]] = {
    "D-dimer":    (0.00, 4.48), "SOFA":       (0.00, 1.48),
    "Lactate":    (0.00, 1.60), "PTX3":       (0.00, 3.70),
    "Urea":       (0.00, 2.30), "IL-8":       (0.00, 4.00),
    "IL-10":      (0.00, 3.70), "PaO2_FiO2":  (1.48, 2.70),
    "sIL-1R2":    (0.00, 4.00), "CRP":        (0.00, 2.70),
    "GCS":        (0.00, 1.20), "IL-6":       (0.00, 5.30),
    "MAP":        (1.15, 2.30), "Platelets":  (1.18, 2.90),
    "IL-18":      (0.00, 3.70), "IL-1ra":     (0.00, 5.70),
    "TNF-alpha":  (0.00, 3.30), "Creatinine": (0.00, 3.30),
}

# Cohort mean energy per marker (used for tipping contribution)
BEI_COHORT_ENERGY_MEANS: Dict[str, float] = {
    "D-dimer":   0.51, "SOFA":      0.48, "Lactate":   0.60,
    "PTX3":      0.54, "Urea":      0.57, "IL-8":      0.59,
    "IL-10":     0.62, "PaO2_FiO2": 0.63, "sIL-1R2":  0.61,
    "CRP":       0.56, "GCS":       0.71, "IL-6":      0.58,
    "MAP":       0.67, "Platelets": 0.70, "IL-18":     0.64,
    "IL-1ra":    0.55, "TNF-alpha": 0.60, "Creatinine": 0.61,
}

# Tipping tier + action per marker
BEI_TIPPING_TIERS: Dict[str, Tuple[str, str]] = {
    # Tier assignments and tipping-associated mortality from Table 2, Chandra 2026 (JAIKE-2026-0015)
    # Tier cutoffs: KILL ≥40%, MODERATE 25–40%, NOISE 20–25%, BENIGN <20%
    # NOTE: No marker fell in the NOISE band in this cohort (Section 4.4)
    # Tipping mortality = 90-day mortality among patients for whom this marker IS the primary tipping dimension
    # ── KILL tier (tipping mortality ≥40%) ──────────────────────────────────────
    "GCS":       ("KILL",     "67% tipping mortality (N=3) | GCS ≤14: 67% overall mortality (16/24); any consciousness impairment triggers HIGH-RISK ALARM"),
    "D-dimer":   ("KILL",     "50% tipping mortality (N=8) | DIC protocol — ISTH scoring, anticoagulation review, coagulation panel q12h"),
    "SOFA":      ("KILL",     "46% tipping mortality (N=13) | Multi-organ support escalation — ventilator, vasopressor, renal replacement review"),
    "PaO2_FiO2": ("KILL",     "45% tipping mortality (N=11) | Respiratory failure — lung-protective ventilation, prone positioning review, FiO2 titration"),
    "Lactate":   ("KILL",     "44% tipping mortality (N=9) | Haemodynamic resuscitation — MAP ≥65 mmHg, vasopressor titration, volume assessment"),
    "PTX3":      ("KILL",     "41% tipping mortality (N=17) | Endothelial collapse — aggressive haemodynamic support, early vasopressor, vascular monitoring"),
    # ── MODERATE tier (tipping mortality 25–40%) ────────────────────────────────
    "IL-8":      ("MODERATE", "33% tipping mortality (N=3) | Neutrophil-driven — antibiotic breadth reassessment, source control review"),
    "Urea":      ("MODERATE", "31% tipping mortality (N=13) | Catabolism/renal — protein delivery ≥1.5 g/kg/day, renal support review, nitrogen balance monitoring"),
    "IL-10":     ("MODERATE", "29% tipping mortality (N=7) | Immunoparalysis — avoid immunosuppressants; persistently elevated IL-10 signals exhaustion, not reserve"),
    # ── BENIGN tier (tipping mortality <20%; includes markers never primary tipping in cohort) ──
    "CRP":       ("BENIGN",   "8% tipping mortality (N=12) | Recoverable acute phase — antibiotic-guided serial monitoring, observe"),
    "MAP":       ("BENIGN",   "0% tipping mortality (N=5) | Maintain haemodynamic targets per protocol"),
    "sIL-1R2":   ("BENIGN",   "0% tipping mortality (N=3) | High entropy weight (0.061) but low tipping mortality; NLRP3 monitoring — serial Day 5 reassessment"),
    "Platelets": ("BENIGN",   "0% tipping mortality (N=1) | Thrombocytopaenia monitoring; secondary to primary coagulopathy"),
    # Not classified — never primary tipping dimension in Chandra 2026 cohort (N=0); tier unassigned
    "IL-1ra":    ("BENIGN",   "N=0 tipping in cohort (unclassified) | Futile counter-regulation — avoid steroid escalation"),
    "TNF-alpha": ("BENIGN",   "N=0 tipping in cohort (unclassified) | Macrophage amplification marker — does not independently drive death in this dataset"),
    "IL-6":      ("BENIGN",   "N=0 tipping in cohort (unclassified) | Prognostically redundant conditional on sIL-1R2, IL-10, PTX3 (Riedemann 2003)"),
    "IL-18":     ("BENIGN",   "N=0 tipping in cohort (unclassified) | Informationally redundant conditional on sIL-1R2 (AUC 0.726 vs 0.610); use sIL-1R2 preferentially"),
    "Creatinine":("BENIGN",   "Not in Chandra 2026 17-marker panel | Use Urea as renal catabolic marker; Creatinine contextual only"),
}

BEI_ZONES: Dict[str, Tuple[float, float, str]] = {
    # Thresholds = 33rd and 67th percentiles of out-of-fold BEI score in analytical cohort
    # Mortality figures = Kaplan-Meier tertile stratification, Fig 10, Chandra 2026 (JAIKE-2026-0015)
    # n=35 per tertile; log-rank p<0.001
    "VERY_LOW":   (0.00, 0.60, "71% 90-day mortality — catastrophic multi-system collapse; all KILL dimensions likely active"),
    "AT_RISK":    (0.60, 0.71, "17% mortality — biologically contested; tipping intervention window is NOW; most actionable zone"),
    "SUFFICIENT": (0.71, 1.00, "14% mortality — energy reserve present; late-mortality phenotype risk (PTX3/sIL-1R2 trajectory)"),
}

GH5_SIGNATURE = {
    "markers":   ["Urea", "IL-10", "sIL-1R2", "PTX3", "SOFA"],
    "auc_5fold": 0.779,
    "auc_loo":   0.800,
    "note": "GH5 primary admission triage instrument (Davoudian 2023).",
}

BEI_PERFORMANCE = {
    # AUC values — Chandra et al. 2026 (JAIKE-2026-0015), Section 4.1
    # Primary figure: out-of-fold 5-fold stratified CV on analytical cohort n=105
    # Complete-case sensitivity (n=33, all 17 markers): AUC=0.885 (95% CI 0.752–0.981)
    "auc_entropy_weighted": 0.793,   # Fig 3B; primary out-of-fold result
    "auc_equal_weighted":   0.799,   # Fig 3B; entropy weighting conferred no advantage over equal-weighted
    "auc_no_lactate":       0.805,   # Section 4.2; 16-marker panel (Lactate excluded for all patients)
    "auc_gh5_5fold":        0.779,   # Section 4.3; Golden Hour Signature 5-fold CV
    "auc_gh5_loo":          0.786,   # Section 4.3; Golden Hour Signature LOO-CV
    # Quintile mortality — Fig 3C; n=21 per quintile; out-of-fold BEI score
    "Q1_mortality": 0.86,   # bottom quintile (lowest biological reserve)
    "Q2_mortality": 0.38,
    "Q3_Q4_mortality": 0.14,  # Section 4.1: "declining to 14% in Q3 and Q4"
    "Q5_mortality": 0.19,   # top quintile; Q5 anomaly discussed in Section 4.2 (Lactate missingness)
    # Zone (tertile) survival — Fig 10 KM; thresholds = 33rd and 67th percentiles of out-of-fold score
    # Very Low (<0.60): 71% mortality  |  At-Risk (0.60–0.71): 17%  |  Sufficient (≥0.71): 14%
    "tier_very_low_mortality":  0.71,  # n=35; Fig 10
    "tier_at_risk_mortality":   0.17,  # n=35; Fig 10
    "tier_sufficient_mortality": 0.14, # n=35; Fig 10
    "cohort_baseline": 0.343,   # 36/105 deaths — analytical cohort mortality rate; Fig 2
    "n_patients": 135, "n_events": 36,
    "cohort": "Chandra et al. 2026, J Artif Intell Knowl Eng (JAIKE-2026-0015); source data: Davoudian et al. 2022, PLOS ONE",
}

# ── Module B — KILL-4 panel ───────────────────────────────────────────────────
# Top 4 markers by Module A entropy weight (KILL tier: all failure markers).
# Raw weights sum = 0.396; renormalized to sum = 1.000.
KILL4_WEIGHTS_RAW: Dict[str, float] = {
    "D-dimer": 0.113, "SOFA": 0.099, "Lactate": 0.095, "PTX3": 0.089,
}
_w4_total = sum(KILL4_WEIGHTS_RAW.values())
KILL4_WEIGHTS: Dict[str, float] = {
    k: round(v / _w4_total, 6) for k, v in KILL4_WEIGHTS_RAW.items()
}
# D-dimer=0.285354  SOFA=0.250000  Lactate=0.239899  PTX3=0.224747

KILL4_PERFORMANCE = {
    "panel":   "KILL-4 (D-dimer, SOFA, Lactate, PTX3)",
    "auc":     None,   # not independently validated — Module A AUC=0.812 as upper bound
    "note": (
        "Module B KILL-4 uses the top-4 KILL-tier markers by Module A entropy weight. "
        "Zone thresholds applied from Module A calibration (Davoudian 2023, N=135). "
        "Module B AUC and thresholds require independent validation before clinical deployment. "
        "Intended for resource-limited settings where cytokine assays are unavailable."
    ),
}


# ── Core computation ──────────────────────────────────────────────────────────

def compute_energy_score(marker_name: str, raw_value: float) -> Optional[float]:
    """
    Convert raw biomarker value to energy score [0,1].
    Pipeline: log10(x+1) → min-max normalise to cohort range → invert if failure marker.
    """
    if raw_value is None:
        return None
    v = float(raw_value)
    if v < 0:
        v = 0.0
    log_val = math.log10(v + 1.0)
    lo, hi = BEI_COHORT_LOG_RANGE.get(marker_name, (0.0, 3.0))
    if hi <= lo:
        return 0.5
    normalised = max(0.0, min(1.0, (log_val - lo) / (hi - lo)))
    energy = (1.0 - normalised) if marker_name in BEI_FAILURE_MARKERS else normalised
    return round(energy, 4)


def compute_bei(
    raw_input: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Compute BEI composite score from a dict of biomarker values.

    weights defaults to BEI_WEIGHTS (Module A, 17 markers).
    Pass KILL4_WEIGHTS to run the Module B (KILL-4) reduced panel.

    Returns full BEI result dict compatible with the v1.1 JSON schema.
    Panel-specific fields:
      panel_size  — number of markers in the weight set
      module      — 'A' (17-marker) or 'B' (KILL-4)
    """
    if weights is None:
        weights = BEI_WEIGHTS

    module_label = "B" if weights is KILL4_WEIGHTS else "A"

    energy_profile: Dict[str, float] = {}
    missing: List[str] = []

    for marker in weights:
        val = raw_input.get(marker)
        if val is None:
            missing.append(marker)
            continue
        score = compute_energy_score(marker, val)
        if score is not None:
            energy_profile[marker] = score
        else:
            missing.append(marker)

    if not energy_profile:
        return {"error": "No valid markers supplied — BEI cannot be computed"}

    avail_weights = {m: weights[m] for m in energy_profile}
    total_weight  = sum(avail_weights.values())
    if total_weight == 0:
        return {"error": "Zero total weight"}

    bei_score = sum(
        energy_profile[m] * avail_weights[m] for m in energy_profile
    ) / total_weight
    bei_score = round(bei_score, 4)

    # Tipping contribution: TC(m) = W(m) × [E(m) − cohort_mean(E(m))]
    # Uses panel weights so TC values sum to zero around the panel's cohort mean
    tipping_contributions = {
        m: round(weights[m] * (e - BEI_COHORT_ENERGY_MEANS.get(m, 0.60)), 5)
        for m, e in energy_profile.items()
    }

    # Most negative TC = primary failing dimension
    tipping_marker = min(tipping_contributions, key=tipping_contributions.get)
    tier_info = BEI_TIPPING_TIERS.get(tipping_marker, ("UNKNOWN", "Assess per protocol"))

    # Zone classification — thresholds = 33rd/67th percentiles of out-of-fold BEI score
    # Source: Chandra 2026 (JAIKE-2026-0015), Fig 10 KM tertile boundaries
    # Module B note: thresholds applied from Module A calibration; not independently validated
    if bei_score < 0.60:
        zone, zone_detail = "VERY_LOW",   BEI_ZONES["VERY_LOW"][2]
    elif bei_score < 0.71:
        zone, zone_detail = "AT_RISK",    BEI_ZONES["AT_RISK"][2]
    else:
        zone, zone_detail = "SUFFICIENT", BEI_ZONES["SUFFICIENT"][2]

    # GCS alarm (only meaningful if GCS was in panel — Module A only)
    gcs_val   = raw_input.get("GCS")
    gcs_alarm = False
    if gcs_val is not None and "GCS" in weights:
        try:
            gcs_alarm = float(gcs_val) <= 14
        except (ValueError, TypeError):
            pass

    # GH5 triage sub-score — only valid for Module A (requires Urea, IL-10, sIL-1R2)
    gh5_markers   = GH5_SIGNATURE["markers"]
    gh5_in_panel  = all(m in weights for m in gh5_markers)
    gh5_avail     = [m for m in gh5_markers if m in energy_profile] if gh5_in_panel else []
    gh5_score     = None
    if gh5_avail:
        w_gh5 = sum(BEI_WEIGHTS[m] for m in gh5_avail)
        gh5_score = round(
            sum(energy_profile[m] * BEI_WEIGHTS[m] for m in gh5_avail) / w_gh5, 4
        )

    panel_size = len(weights)

    return {
        "bei_score":               bei_score,
        "bei_zone":                zone,
        "bei_zone_detail":         zone_detail,
        "energy_profile":          energy_profile,
        "tipping_contributions":   tipping_contributions,
        "tipping_marker":          tipping_marker,
        "tipping_tier":            tier_info[0],
        "tipping_action":          tier_info[1],
        "gcs_alarm":               gcs_alarm,
        "gh5_score":               gh5_score,
        "missing_markers":         missing,
        "n_available":             len(energy_profile),
        "n_missing":               len(missing),
        "lactate_missing_warning": "Lactate" in missing,
        "coverage_pct":            round(len(energy_profile) / panel_size * 100, 1),
        "panel_size":              panel_size,
        "module":                  module_label,
    }


def compute_bei_b(raw_input: Dict[str, Optional[float]]) -> dict:
    """
    Module B — KILL-4 panel (D-dimer, SOFA, Lactate, PTX3).

    Uses renormalized weights summing to 1.0. GH5 is suppressed (incomplete panel).
    Zone thresholds from Module A calibration — not independently validated.
    AUC not independently validated (Module A AUC=0.812 is the upper-bound reference).
    """
    return compute_bei(raw_input, weights=KILL4_WEIGHTS)


def compare_modules(bei_a: dict, bei_b: dict) -> dict:
    """
    Structured comparison of Module A (17-marker) vs Module B (KILL-4) results.

    Intended to support accuracy / prediction cross-validation analysis.
    Zone agreement uses Module A calibration thresholds for both modules
    (Module B thresholds not independently validated).
    """
    zone_agreement  = bei_a["bei_zone"] == bei_b["bei_zone"]
    score_delta     = round(bei_a["bei_score"] - bei_b["bei_score"], 4)
    tip_agreement   = bei_a.get("tipping_marker", "") == bei_b.get("tipping_marker", "")

    if abs(score_delta) < 0.02:
        delta_note = "Scores concordant (|Δ| < 0.02)"
    elif score_delta > 0:
        delta_note = (
            f"Module A scores {score_delta:+.4f} higher than Module B "
            f"— KILL-4 panel may underestimate biological reserve "
            f"(missing protective markers: GCS, MAP, PaO2/FiO2, Platelets)"
        )
    else:
        delta_note = (
            f"Module B scores {-score_delta:.4f} higher than Module A "
            f"— KILL-4 panel may overestimate biological reserve "
            f"(missing immunological burden: IL-8, IL-10, Urea, CRP, sIL-1R2, etc.)"
        )

    return {
        "zone_agreement":      zone_agreement,
        "score_delta":         score_delta,
        "delta_note":          delta_note,
        "tipping_agreement":   tip_agreement,
        "module_a_score":      bei_a["bei_score"],
        "module_b_score":      bei_b["bei_score"],
        "module_a_zone":       bei_a["bei_zone"],
        "module_b_zone":       bei_b["bei_zone"],
        "module_a_tipping":    bei_a.get("tipping_marker", ""),
        "module_b_tipping":    bei_b.get("tipping_marker", ""),
        "module_a_coverage":   bei_a.get("coverage_pct", 0.0),
        "module_b_coverage":   bei_b.get("coverage_pct", 0.0),
        "validation_note": (
            "Module B zone thresholds applied from Module A calibration (Davoudian 2023). "
            "Module B AUC is unvalidated. "
            "For research use only — independent validation cohort required "
            "before Module B can be used standalone in clinical decision support."
        ),
    }


def format_bei_context(bei: dict, max_chars: int = 500) -> str:
    """Human-readable BEI summary for prompt injection."""
    if "error" in bei:
        return f"BEI COMPUTATION ERROR: {bei['error']}"

    module     = bei.get("module", "A")
    panel_size = bei.get("panel_size", len(BEI_WEIGHTS))

    zone_label = {
        "VERY_LOW":   "⚠ VERY_LOW — catastrophic collapse",
        "AT_RISK":    "⚠ AT_RISK — intervention window",
        "SUFFICIENT": "✓ SUFFICIENT — energy reserve present",
    }.get(bei["bei_zone"], bei["bei_zone"])

    if module == "B":
        header = "═══ BEI-B KILL-4 Module (Davoudian 2023 weights; AUC pending validation) ═══"
    else:
        header = "═══ BEI v4 Module A (Davoudian 2023, N=135, AUC=0.812) ═══"

    lines = [
        header,
        f"BEI Score : {bei['bei_score']:.4f}  [0=collapse, 1=homeostasis]",
        f"Zone      : {zone_label}",
        f"           {bei['bei_zone_detail']}",
        "",
        f"PRIMARY TIPPING DIMENSION: {bei['tipping_marker']}",
        f"  Tier   : {bei['tipping_tier']}-tier",
        f"  Action : {bei['tipping_action']}",
        f"  TC     : {bei['tipping_contributions'].get(bei['tipping_marker'],0):+.5f}",
        "",
    ]

    neg_tcs = sorted(
        [(m, tc) for m, tc in bei["tipping_contributions"].items() if tc < 0],
        key=lambda x: x[1]
    )[:3]
    if neg_tcs:
        lines.append("TOP 3 FAILING DIMENSIONS:")
        for m, tc in neg_tcs:
            tier = BEI_TIPPING_TIERS.get(m, ("?",))[0]
            e    = bei["energy_profile"].get(m, 0.0)
            lines.append(f"  {m}: TC={tc:+.5f}, Energy={e:.3f} [{tier}-tier]")
        lines.append("")

    if bei.get("gh5_score") is not None:
        lines.append(f"GH5 Triage Sub-score: {bei['gh5_score']:.4f}")
    if bei.get("gcs_alarm"):
        lines.append("⚠ GCS ALARM: GCS ≤14 — 67% 90-day mortality")
    if bei.get("lactate_missing_warning"):
        lines.append("⚠ LACTATE MISSING: MNAR bias — BEI likely artificially inflated")
    if module == "B":
        lines.append("⚠ MODULE B: Zone thresholds from Module A calibration — not independently validated")

    lines.append(
        f"Coverage: {bei['n_available']}/{panel_size} markers "
        f"({bei['coverage_pct']:.0f}%) | Missing: "
        f"{', '.join(bei['missing_markers']) if bei['missing_markers'] else 'None'}"
    )
    return "\n".join(lines)


def build_clinical_flags(bei: dict, raw: Dict[str, Optional[float]]) -> List[str]:
    """Build clinical alert flags from BEI result and raw input."""
    flags: List[str] = []

    if bei.get("gcs_alarm"):
        flags.append("GCS ALARM: GCS ≤14 — 67% 90-day mortality threshold")

    if bei.get("lactate_missing_warning"):
        flags.append("LACTATE MISSING — BEI may be inflated (MNAR bias)")

    # U:Cr ratio flag
    urea  = raw.get("Urea", 0) or 0
    creat = raw.get("Creatinine", 1) or 1
    # Unit guard: creatinine < 20 μmol/L is biologically impossible; likely mg/dL entry
    if 0 < creat < 20:
        creat = round(creat * 88.4, 1)
    creat_si = creat / 1000.0
    if creat_si > 0:
        uc = round(urea / creat_si, 1)
        if uc > 150:
            flags.append(
                f"U:Cr ratio={uc} (>150) — CATABOLIC HYPERUREAEMIA dominant, not primary AKI"
            )

    # IL-10 steroid flag
    il10_e = bei.get("energy_profile", {}).get("IL-10", 0.0)
    if il10_e > 0.70:
        flags.append(
            f"IL-10 Energy={il10_e:.4f} (>0.70) — Corticosteroids CONTRAINDICATED"
        )

    zone = bei.get("bei_zone", "")
    flags.append(
        f"BEI Zone: {zone} | BEI={bei.get('bei_score',0):.4f} | "
        f"Tipping: {bei.get('tipping_marker','?')} [{bei.get('tipping_tier','?')}-tier]"
    )

    return flags
