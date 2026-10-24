"""
test_bei.py — Standalone BEI + xAI test (no database, no API server required)
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Usage:
    py -3.11 test_bei.py
"""
from bei_engine import compute_bei
from xai_engine import compute_xai_attribution, format_xai_for_prompt

# ── Test patient: severe sepsis profile ──────────────────────────────────────
patient = {
    "Lactate":    4.2,    # mmol/L  — elevated (normal <2.0)
    "SOFA":       11,     # score   — severe (normal <2)
    "PTX3":       28.8,   # ng/mL   — highly elevated (normal <2)
    "D-dimer":    3200,   # ng/mL   — elevated (normal <500)
    "IL-10":      3.2,    # pg/mL
    "Creatinine": 180,    # μmol/L  — elevated
    "Urea":       12.0,   # mmol/L
    "Platelets":  95,     # ×10⁹/L  — low
    "MAP":        62,     # mmHg    — borderline
    "GCS":        14,     # score   — borderline alarm
    "CRP":        210,    # mg/L    — very elevated
    "PaO2_FiO2":  180,    # ratio   — impaired oxygenation
}

print("=" * 60)
print("SepsisAI BEI — Standalone Test")
print("INVESTIGATIONAL — Not FDA Cleared")
print("=" * 60)

# ── Run BEI ──────────────────────────────────────────────────────────────────
print("\n[1/2] Computing BEI score...")
result = compute_bei(patient)

print(f"\n  BEI Score   : {result['bei_score']:.4f}")
print(f"  Zone        : {result['bei_zone']}")
print(f"  Zone Detail : {result['bei_zone_detail']}")
print(f"  Tipping     : {result['tipping_marker']} [{result['tipping_tier']} tier]")
print(f"  GCS Alarm   : {result['gcs_alarm']}")
print(f"  Coverage    : {result['coverage_pct']:.0f}% ({result['n_available']}/{result['n_available']+result['n_missing']} markers)")

print("\n  Top tipping contributions (most negative = primary failure):")
tc = result["tipping_contributions"]
ep = result["energy_profile"]
sorted_tc = sorted(tc.items(), key=lambda x: x[1])[:5]
for marker, val in sorted_tc:
    energy = ep.get(marker, 0)
    print(f"    {marker:15s}  TC={val:+.5f}  Energy={energy:.4f}")

# ── Run xAI ──────────────────────────────────────────────────────────────────
print("\n[2/2] Computing xAI attribution...")
xai = compute_xai_attribution(
    bei_result=result,
    raw_input=patient,
    module="A",
)

print(f"\n  Predicted outcome : {xai['predicted_outcome']}")
print(f"  Zone confidence   : {xai['zone_confidence']}")
print(f"  Summary           : {xai['xai_summary']}")

print("\n  xAI Attribution Table:")
print(format_xai_for_prompt(xai["xai_attribution"]))

print("\n" + "=" * 60)
print("Test complete. No database or API server required.")
print("=" * 60)
