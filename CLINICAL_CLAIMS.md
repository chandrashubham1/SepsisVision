# CLINICAL_CLAIMS.md — BEI Clinical Claims Registry

**Purpose**: Paper trail for every quantitative clinical claim made by this tool.
Each claim is traced to a specific figure, table, or section in the primary reference.
Any change to an allowed value requires updating this file and committing with the
`clinical(C-XXX): ...` message format referencing the relevant source.

**Primary reference**: Chandra et al. 2026, *J Artif Intell Knowl Eng* (JAIKE-2026-0015).
DOI: 10.37819/Jaike.26.0015  
Source cohort: Davoudian et al. 2022, PLOS ONE (N=135 patients, 105 with confirmed 90-day outcome)

---

## Legend

| Column | Meaning |
|--------|---------|
| **Claim ID** | Unique reference used in commit messages |
| **Claim text** | Exact claim the tool makes (or should make) |
| **Source** | Figure/table/section in Chandra 2026 |
| **Allowed value** | Value the code must use |
| **Code location** | File and constant/line |
| **Status** | ✅ Calibrated / ⚠ Needs validation / ❌ Wrong |

---

## Section 1 — Model Performance (AUC)

| Claim ID | Claim text | Source | Allowed value | Code location | Status |
|----------|-----------|--------|--------------|---------------|--------|
| C-001 | Primary BEI AUC (entropy-weighted, Module A) | Fig 3B; out-of-fold 5-fold CV, n=105 | **0.793** | `bei_engine.py` `BEI_PERFORMANCE["auc_entropy_weighted"]` | ✅ |
| C-002 | Equal-weighted BEI AUC | Fig 3B | **0.799** | `bei_engine.py` `BEI_PERFORMANCE["auc_equal_weighted"]` | ✅ |
| C-003 | No-lactate BEI AUC (16-marker) | Section 4.2 | **0.805** | `bei_engine.py` `BEI_PERFORMANCE["auc_no_lactate"]` | ✅ |
| C-004 | GH5 signature AUC (5-fold CV) | Section 4.3 | **0.779** | `bei_engine.py` `BEI_PERFORMANCE["auc_gh5_5fold"]` | ✅ |
| C-005 | GH5 signature AUC (LOO-CV) | Section 4.3 | **0.786** | `bei_engine.py` `BEI_PERFORMANCE["auc_gh5_loo"]` | ✅ |
| C-006 | Module B (KILL-4) AUC | Not independently validated in Chandra 2026 | **None** (use Module A as upper bound) | `bei_engine.py` `KILL4_PERFORMANCE["auc"]` | ✅ |

> ⚠ **C-006 pending**: User reports KILL-4 may have better prediction accuracy. Once AUC data is available, create C-006a with exact value, cohort, and validation method.

---

## Section 2 — Cohort Metadata

| Claim ID | Claim text | Source | Allowed value | Code location | Status |
|----------|-----------|--------|--------------|---------------|--------|
| C-010 | Total cohort N | Abstract / Methods | **N=135** | `bei_engine.py` `BEI_PERFORMANCE["n_patients"]` | ✅ |
| C-011 | Number of 90-day deaths | Fig 3C / Methods | **36** | `bei_engine.py` `BEI_PERFORMANCE["n_events"]` | ✅ |
| C-012 | Analytical cohort (confirmed outcome) | Methods | **n=105** | Referenced in zone/quintile mortality claims | ✅ |
| C-013 | Cohort baseline 90-day mortality | 36/105 deaths in analytical cohort | **34.3%** | `bei_engine.py` `BEI_PERFORMANCE["cohort_baseline"]` = 0.343 | ✅ |

---

## Section 3 — Quintile Mortality (Fig 3C)

Each quintile = 21 patients; out-of-fold BEI score used.

| Claim ID | Claim text | Source | Allowed value | Code location | Status |
|----------|-----------|--------|--------------|---------------|--------|
| C-020 | Q1 (lowest BEI) 90-day mortality | Fig 3C | **86%** | `bei_engine.py` `BEI_PERFORMANCE["Q1_mortality"]` = 0.86 | ✅ |
| C-021 | Q2 90-day mortality | Fig 3C | **38%** | `bei_engine.py` `BEI_PERFORMANCE["Q2_mortality"]` = 0.38 | ✅ |
| C-022 | Q3 and Q4 90-day mortality | Fig 3C ("declining to 14% in Q3 and Q4") | **14%** | `bei_engine.py` `BEI_PERFORMANCE["Q3_Q4_mortality"]` = 0.14 | ✅ |
| C-023 | Q5 (highest BEI) 90-day mortality | Fig 3C | **19%** | `bei_engine.py` `BEI_PERFORMANCE["Q5_mortality"]` = 0.19 | ✅ |

---

## Section 4 — Zone (Tertile) Mortality and Thresholds (Fig 10)

Thresholds = 33rd and 67th percentiles of out-of-fold BEI score; n=35 per tertile; log-rank p<0.001.

| Claim ID | Claim text | Source | Allowed value | Code location | Status |
|----------|-----------|--------|--------------|---------------|--------|
| C-030 | VERY_LOW zone lower threshold | Fig 10 (33rd pctile of out-of-fold BEI) | **0.00** | `bei_engine.py` `BEI_ZONES["VERY_LOW"][0]` | ✅ |
| C-031 | VERY_LOW zone upper threshold | Fig 10 | **0.60** | `bei_engine.py` `BEI_ZONES["VERY_LOW"][1]`; `compute_bei()` branch | ✅ |
| C-032 | AT_RISK zone upper threshold | Fig 10 (67th pctile) | **0.71** | `bei_engine.py` `BEI_ZONES["AT_RISK"][1]`; `compute_bei()` branch | ✅ |
| C-033 | VERY_LOW zone 90-day mortality (KM tertile) | Fig 10 | **71%** | `bei_engine.py` `BEI_PERFORMANCE["tier_very_low_mortality"]` = 0.71 | ✅ |
| C-034 | AT_RISK zone 90-day mortality (KM tertile) | Fig 10 | **17%** | `bei_engine.py` `BEI_PERFORMANCE["tier_at_risk_mortality"]` = 0.17 | ✅ |
| C-035 | SUFFICIENT zone 90-day mortality (KM tertile) | Fig 10 | **14%** | `bei_engine.py` `BEI_PERFORMANCE["tier_sufficient_mortality"]` = 0.14 | ✅ |

---

## Section 5 — Tipping Tier Assignments (Table 2)

Tipping mortality = 90-day mortality among patients where this marker is the **primary tipping dimension**.  
Tier cutoffs: KILL ≥40%, MODERATE 25–40%, NOISE 20–25%, BENIGN <20%.  
**Note**: No marker fell in the NOISE band in this cohort (Section 4.4 / Table 2 footnote).

| Claim ID | Marker | Tier | Tipping mortality | N | Source | Code location | Status |
|----------|--------|------|-------------------|---|--------|---------------|--------|
| C-050 | GCS | KILL | 67% | 3 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["GCS"]` | ✅ |
| C-051 | D-dimer | KILL | 50% | 8 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["D-dimer"]` | ✅ |
| C-052 | SOFA | KILL | 46% | 13 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["SOFA"]` | ✅ |
| C-053 | PaO2/FiO2 | KILL | 45% | 11 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["PaO2_FiO2"]` | ✅ |
| C-054 | Lactate | KILL | 44% | 9 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["Lactate"]` | ✅ |
| C-055 | PTX3 | KILL | 41% | 17 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["PTX3"]` | ✅ |
| C-056 | IL-8 | MODERATE | 33% | 3 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["IL-8"]` | ✅ |
| C-057 | Urea | MODERATE | 31% | 13 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["Urea"]` | ✅ |
| C-058 | IL-10 | MODERATE | 29% | 7 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["IL-10"]` | ✅ |
| C-059 | CRP | BENIGN | 8% | 12 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["CRP"]` | ✅ |
| C-060 | MAP | BENIGN | 0% | 5 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["MAP"]` | ✅ |
| C-061 | sIL-1R2 | BENIGN | 0% | 3 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["sIL-1R2"]` | ✅ |
| C-062 | Platelets | BENIGN | 0% | 1 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["Platelets"]` | ✅ |
| C-063 | IL-1ra | unclassified (N=0) | — | 0 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["IL-1ra"]` (BENIGN placeholder) | ✅ |
| C-064 | TNF-α | unclassified (N=0) | — | 0 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["TNF-alpha"]` | ✅ |
| C-065 | IL-6 | unclassified (N=0) | — | 0 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["IL-6"]` | ✅ |
| C-066 | IL-18 | unclassified (N=0) | — | 0 | Table 2 | `bei_engine.py` `BEI_TIPPING_TIERS["IL-18"]` | ✅ |

---

## Section 6 — Narrative Claims (Grounding Constraints)

| Claim ID | Claim text | Source | Allowed value | Code location | Status |
|----------|-----------|--------|--------------|---------------|--------|
| C-070 | Module A marker count | Methods | **17** | `xai_engine.py` `build_rounds_prompt()` grounding constraint | ✅ |
| C-071 | Module B (KILL-4) marker count | Methods | **4** | `xai_engine.py` `build_rounds_prompt()` grounding constraint | ✅ |
| C-072 | Lactate elevated threshold | Sepsis-3 (Singer et al. 2016, JAMA) | **≥2.0 mmol/L** | `xai_engine.py` grounding constraint | ✅ |
| C-073 | Lactate high-risk threshold | Sepsis-3 | **≥4.0 mmol/L** | `xai_engine.py` grounding constraint | ✅ |
| C-074 | IL-10 energy steroid-CI threshold | Hard rule R2, `xai_engine.py` | **>0.70** | `xai_engine.py` R2 rule; grounding constraint | ✅ |

---

## Section 7 — Open / Unvalidated Claims

| Claim ID | Description | Action required |
|----------|-------------|-----------------|
| C-080 | KILL-4 (Module B) prediction accuracy claim | User states KILL-4 has better accuracy than Module A — requires AUC value, cohort size, CV method, and reference. Add as C-006a once data provided. |
| C-081 | Module B zone thresholds | Applied from Module A calibration (33rd/67th pctile of Module A out-of-fold scores). Not independently validated. Marked in code as unvalidated. |
| C-082 | GH5 Golden Hour Signature clinical utility | AUC validated (C-004/C-005) but clinical deployment threshold not established. |

---

## Change Log

| Date | Commit | Claims affected | Change summary |
|------|--------|----------------|----------------|
| 2026-10-25 | (pending) | C-001 to C-074 | Initial calibration against Chandra 2026 (JAIKE-2026-0015). Corrected AUC, quintile mortality, zone thresholds, tipping tier assignments. Added U:Cr unit guard. Added narrative grounding constraints. |

---

*Last updated: 2026-10-25 — Calibration session against Chandra et al. 2026 (JAIKE-2026-0015)*
