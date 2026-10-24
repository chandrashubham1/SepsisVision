"""
test_bei.py — Standalone BEI Module A + B test (no database, no API server required)
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Usage:
    py -3.11 test_bei.py
"""
from bei_engine import compute_bei, compute_bei_b, compare_modules
from xai_engine import build_xai_attribution, format_xai_for_prompt

# ── Test patient: severe sepsis profile ──────────────────────────────────────
patient = {
    # Module A markers (17-marker full panel)
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

# Module B only needs the KILL-4 markers
kill4 = {
    "D-dimer": patient["D-dimer"],
    "SOFA":    patient["SOFA"],
    "Lactate": patient["Lactate"],
    "PTX3":    patient["PTX3"],
}

print("=" * 65)
print("SepsisAI BEI — Standalone Test: Module A (17-marker) + Module B (KILL-4)")
print("INVESTIGATIONAL — Not FDA Cleared")
print("=" * 65)

# ── Module A ─────────────────────────────────────────────────────────────────
print("\n[1/3] Module A — Full 17-marker BEI")
print("-" * 40)
a = compute_bei(patient)
print(f"  BEI Score   : {a['bei_score']:.4f}")
print(f"  Zone        : {a['bei_zone']}")
print(f"  Tipping     : {a['tipping_marker']} [{a['tipping_tier']} tier]")
print(f"  GCS Alarm   : {a['gcs_alarm']}")
print(f"  Coverage    : {a['coverage_pct']:.0f}% ({a['n_available']} markers present)")
print(f"  Zone Detail : {a['bei_zone_detail']}")

print("\n  Top 5 tipping contributions:")
tc_a = a["tipping_contributions"]
ep_a = a["energy_profile"]
for marker, val in sorted(tc_a.items(), key=lambda x: x[1])[:5]:
    print(f"    {marker:15s}  TC={val:+.5f}  Energy={ep_a.get(marker,0):.4f}")

# ── Module B ─────────────────────────────────────────────────────────────────
print("\n[2/3] Module B — KILL-4 reduced panel (D-dimer, SOFA, Lactate, PTX3)")
print("-" * 40)
b = compute_bei_b(kill4)
print(f"  BEI Score   : {b['bei_score']:.4f}")
print(f"  Zone        : {b['bei_zone']}")
print(f"  Tipping     : {b['tipping_marker']} [{b['tipping_tier']} tier]")
print(f"  Coverage    : {b['coverage_pct']:.0f}% ({b['n_available']} markers present)")
print(f"  Zone Detail : {b['bei_zone_detail']}")

print("\n  KILL-4 tipping contributions:")
tc_b = b["tipping_contributions"]
ep_b = b["energy_profile"]
for marker, val in sorted(tc_b.items(), key=lambda x: x[1]):
    print(f"    {marker:15s}  TC={val:+.5f}  Energy={ep_b.get(marker,0):.4f}")

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n[3/3] Module A vs B Comparison")
print("-" * 40)
comp = compare_modules(a, b)
print(f"  Zone agreement    : {comp['zone_agreement']}")
print(f"  Score delta (A-B) : {comp['score_delta']:+.4f}")
print(f"  Tipping agreement : {comp['tipping_agreement']}")
print(f"  Module A          : {comp['module_a_zone']} ({comp['module_a_score']:.4f})  tipping={comp['module_a_tipping']}")
print(f"  Module B          : {comp['module_b_zone']} ({comp['module_b_score']:.4f})  tipping={comp['module_b_tipping']}")
print(f"  Note              : {comp['delta_note']}")
print(f"  Validation note   : {comp['validation_note']}")

# ── xAI for Module A ─────────────────────────────────────────────────────────
print("\n[xAI] Module A attribution table")
print("-" * 40)
xai = build_xai_attribution(bei=a, raw=patient)
print(f"  {xai['xai_summary']}")
print()
print(format_xai_for_prompt(xai))

print("\n" + "=" * 65)
print("Test complete.")
print("=" * 65)
