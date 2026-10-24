"""
lab_parser.py — Blood report PDF parser for SepsisAI API
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Handles two report types:
  1. Standard clinical labs (CBC, CMP, ABG, coagulation) — most hospital PDFs
  2. Cytokine panels (Luminex / ELISA) — research/specialty lab PDFs

Strategy:
  - Extract text with PyMuPDF (fitz)
  - Apply marker-specific regex patterns for value extraction
  - Return dict keyed by canonical BEI marker names
  - Report which markers were found vs. missing
  - Apply Davoudian 2023 unit corrections where needed
"""

from __future__ import annotations
import io
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# All 17 BEI markers in canonical form
ALL_BEI_MARKERS = [
    "D-dimer", "SOFA", "Lactate", "PTX3", "Urea", "IL-8", "IL-10",
    "GCS", "sIL-1R2", "CRP", "PaO2_FiO2", "IL-6", "MAP",
    "Platelets", "IL-18", "IL-1ra", "TNF-alpha", "Creatinine",
]

# ── Regex patterns per marker ──────────────────────────────────────────────────
# Each pattern: (canonical_name, [list of patterns], unit_hint)
# Patterns try common lab report phrasings; capture group 1 = numeric value.

_FLOAT = r"[-+]?\d+(?:[.,]\d+)?"  # number with optional decimal

MARKER_PATTERNS: List[Tuple[str, List[str], str]] = [
    ("Lactate", [
        rf"[Ll]actate\s*[:\(]?\s*({_FLOAT})\s*(?:mmol/[lL]|mM)",
        rf"[Ll]actic\s+[Aa]cid\s*[:\(]?\s*({_FLOAT})\s*(?:mmol/[lL]|mM)",
        rf"[Ll]AC\s*[:\(]?\s*({_FLOAT})\s*(?:mmol/[lL]|mM)",
    ], "mmol/L"),

    ("Creatinine", [
        rf"[Cc]reatinine\s*[:\(]?\s*({_FLOAT})\s*(?:μmol/[lL]|umol/[lL]|µmol/[lL])",
        rf"[Cc]reatinine\s*[:\(]?\s*({_FLOAT})\s*(?:mg/dL)",      # will convert
        rf"\bCREA\b\s*[:\(]?\s*({_FLOAT})\s*(?:μmol|umol|µmol)",
    ], "μmol/L"),

    ("Urea", [
        rf"[Uu]rea\s*[:\(]?\s*({_FLOAT})\s*(?:mmol/[lL]|mM)",
        rf"[Bb][Uu][Nn]\s*[:\(]?\s*({_FLOAT})\s*(?:mg/dL)",       # BUN → convert ×0.357
        rf"[Bb]lood\s+[Uu]rea\s+[Nn]itrogen\s*[:\(]?\s*({_FLOAT})\s*(?:mg/dL)",
        rf"[Uu]rea\s*[:\(]?\s*({_FLOAT})\s*(?:mg/dL)",            # will convert
    ], "mmol/L"),

    ("Platelets", [
        rf"[Pp]latelets?\s*[:\(]?\s*({_FLOAT})\s*(?:×10[³3]|x10[³3]|10[³3]|K/[μu]L|/μL|/uL)",
        rf"[Pp]LT\s*[:\(]?\s*({_FLOAT})\s*(?:×10[³3]|x10[³3]|K/[μu]L|/μL)",
        rf"[Pp]latelets?\s*[:\(]?\s*({_FLOAT})\s*(?:\d{3}(?:,\d{3})?)",   # raw count
    ], "×10⁹/L"),

    ("CRP", [
        rf"[Cc][-. ]?[Rr]eactive\s+[Pp]rotein\s*[:\(]?\s*({_FLOAT})\s*(?:mg/[lL])",
        rf"\bCRP\b\s*[:\(]?\s*({_FLOAT})\s*(?:mg/[lL])",
        rf"[Cc][-. ]?[Rr]eactive\s+[Pp]rotein\s*[:\(]?\s*({_FLOAT})",
    ], "mg/L"),

    ("MAP", [
        rf"[Mm]ean\s+[Aa]rterial\s+[Pp]ressure\s*[:\(]?\s*({_FLOAT})\s*(?:mmHg)?",
        rf"\bMAP\b\s*[:\(]?\s*({_FLOAT})\s*(?:mmHg)?",
    ], "mmHg"),

    ("GCS", [
        rf"[Gg]lasgow\s+[Cc]oma\s+[Ss]cale?\s*[:\(]?\s*({_FLOAT})",
        rf"\bGCS\b\s*(?:score|=|:)?\s*[:\(]?\s*({_FLOAT})",
        rf"[Gg]lasgow\s+[Cc]oma\s+[Ss]core\s*[:\(]?\s*({_FLOAT})",
    ], "score"),

    # PaO2/FiO2 — most lab reports separate PaO2 and FiO2 on ABG
    ("PaO2_FiO2", [
        rf"[Pp][Aa][Oo]2\s*/\s*[Ff][Ii][Oo]2\s*[:\(]?\s*({_FLOAT})",
        rf"P/F\s+ratio\s*[:\(]?\s*({_FLOAT})",
        rf"[Hh]orowitz\s*(?:index|ratio)?\s*[:\(]?\s*({_FLOAT})",
    ], "ratio×100"),

    ("D-dimer", [
        rf"[Dd]-?[Dd]imer\s*[:\(]?\s*({_FLOAT})\s*(?:ng/m[lL]|ng/mL)",
        rf"[Dd]-?[Dd]imer\s*[:\(]?\s*({_FLOAT})\s*(?:μg/m[lL]|ug/m[lL])",  # convert ×1000
        rf"[Dd]-?[Dd]imer\s*[:\(]?\s*({_FLOAT})\s*(?:mg/[lL])",              # convert ×1000
    ], "ng/mL"),

    # SOFA — often computed separately; try to find it
    ("SOFA", [
        rf"[Ss][Oo][Ff][Aa]\s+(?:[Ss]core)?\s*[:\(=]?\s*({_FLOAT})",
        rf"Sequential\s+Organ\s+Failure\s+Assessment\s*[:\(]?\s*({_FLOAT})",
    ], "score"),

    # Cytokines — Luminex/ELISA panels
    ("PTX3", [
        rf"[Pp][Tt][Xx]-?3\s*[:\(]?\s*({_FLOAT})\s*(?:ng/m[lL]|ng/mL)",
        rf"[Pp]entraxin[-\s]?3\s*[:\(]?\s*({_FLOAT})\s*(?:ng/m[lL])",
    ], "ng/mL"),

    ("sIL-1R2", [
        rf"s[Ii][Ll]-?1[Rr]2\s*[:\(]?\s*({_FLOAT})\s*(?:ng/m[lL]|ng/mL)",
        rf"sIL1R2\s*[:\(]?\s*({_FLOAT})\s*(?:ng/m[lL])",
        rf"[Ss]oluble\s+IL-?1R2\s*[:\(]?\s*({_FLOAT})",
    ], "ng/mL"),

    ("IL-10", [
        rf"[Ii][Ll]-?10\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"[Ii]nterleukin[-\s]?10\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL])",
    ], "pg/mL"),

    ("IL-8", [
        rf"[Ii][Ll]-?8\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"CXCL8\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL])",
        rf"[Ii]nterleukin[-\s]?8\s*[:\(]?\s*({_FLOAT})",
    ], "pg/mL"),

    ("IL-6", [
        rf"[Ii][Ll]-?6\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"[Ii]nterleukin[-\s]?6\s*[:\(]?\s*({_FLOAT})",
    ], "pg/mL"),

    ("IL-18", [
        rf"[Ii][Ll]-?18\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"[Ii]nterleukin[-\s]?18\s*[:\(]?\s*({_FLOAT})",
    ], "pg/mL"),

    ("IL-1ra", [
        rf"[Ii][Ll]-?1[Rr][Aa]\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"[Ii][Ll][-\s]?1\s+[Rr]eceptor\s+[Aa]ntagonist\s*[:\(]?\s*({_FLOAT})",
    ], "pg/mL"),

    ("TNF-alpha", [
        rf"[Tt][Nn][Ff]-?(?:alpha|α)\s*[:\(]?\s*({_FLOAT})\s*(?:pg/m[lL]|pg/mL)",
        rf"[Tt]umou?r\s+[Nn]ecrosis\s+[Ff]actor\s*[:\(]?\s*({_FLOAT})",
    ], "pg/mL"),
]


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except (ValueError, TypeError):
        return None


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except ImportError:
        raise RuntimeError("PyMuPDF not installed — run: pip install pymupdf")
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")


def _detect_patient_id(text: str) -> Optional[str]:
    """Try to extract patient ID or MRN from report text."""
    patterns = [
        r"(?:Patient\s+ID|MRN|Medical\s+Record|Patient\s+#)[:\s]+([A-Z0-9\-_]{4,20})",
        r"(?:Patient|Name)[:\s]+([A-Z][a-z]+,?\s+[A-Z][a-z]+)",
        r"(?:ID|Accession)[:\s#]+([A-Z0-9\-]{5,20})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _apply_unit_corrections(
    marker: str,
    value: float,
    matched_text: str,
) -> Tuple[float, Optional[str]]:
    """
    Apply Davoudian 2023 unit corrections where the regex matched a non-SI unit.
    Returns (corrected_value, correction_note or None).
    """
    note = None

    # Creatinine: mg/dL → μmol/L (× 88.42)
    if marker == "Creatinine" and "mg/dL" in matched_text:
        corrected = round(value * 88.42, 2)
        note = f"Creatinine: {value} mg/dL × 88.42 → {corrected} μmol/L"
        return corrected, note

    # Urea: mg/dL → mmol/L (÷ 6.006) [BUN mg/dL → mmol/L × 0.357]
    if marker == "Urea" and "mg/dL" in matched_text:
        corrected = round(value / 6.006, 2)
        note = f"Urea: {value} mg/dL ÷ 6.006 → {corrected} mmol/L"
        return corrected, note

    # BUN mg/dL → Urea mmol/L (× 0.357)
    if marker == "Urea" and "BUN" in matched_text.upper():
        corrected = round(value * 0.357, 2)
        note = f"BUN: {value} mg/dL × 0.357 → {corrected} mmol/L Urea"
        return corrected, note

    # D-dimer: μg/mL or mg/L → ng/mL (× 1000)
    if marker == "D-dimer" and any(u in matched_text for u in ["μg/mL", "ug/mL", "mg/L"]):
        corrected = round(value * 1000.0, 2)
        note = f"D-dimer: {value} μg/mL × 1000 → {corrected} ng/mL"
        return corrected, note

    # Platelets: raw count (>10000) → ×10⁹/L (÷ 1000)
    if marker == "Platelets" and value > 10000:
        corrected = round(value / 1000.0, 1)
        note = f"Platelets: {value} → {corrected} ×10⁹/L (÷1000)"
        return corrected, note

    # PaO2/FiO2: if extracted value < 10 it's probably the raw decimal ratio (× 100)
    if marker == "PaO2_FiO2" and value < 10:
        corrected = round(value * 100.0, 1)
        note = f"PaO2/FiO2: {value} (decimal) × 100 → {corrected}"
        return corrected, note

    return value, note


def _try_pao2_fio2_from_abg(text: str) -> Optional[Tuple[float, str]]:
    """
    If PaO2/FiO2 ratio not directly stated, compute from ABG PaO2 + FiO2.
    Returns (computed_ratio, note) or None.
    """
    pao2_pat = rf"[Pp][Aa][Oo]2\s*[:\(]?\s*({_FLOAT})\s*(?:mmHg|kPa)?"
    fio2_pat = rf"[Ff][Ii][Oo]2\s*[:\(]?\s*({_FLOAT})"

    pao2_m = re.search(pao2_pat, text)
    fio2_m = re.search(fio2_pat, text)
    if not (pao2_m and fio2_m):
        return None

    pao2 = _safe_float(pao2_m.group(1))
    fio2 = _safe_float(fio2_m.group(1))
    if pao2 is None or fio2 is None or fio2 == 0:
        return None

    # FiO2 may be expressed as fraction (0.21) or percentage (21)
    if fio2 > 1.5:
        fio2 = fio2 / 100.0

    ratio = round(pao2 / fio2, 1)
    note  = f"PaO2/FiO2: PaO2={pao2} mmHg / FiO2={fio2:.2f} = {ratio}"
    return ratio, note


def parse_blood_report_pdf(pdf_bytes: bytes) -> dict:
    """
    Parse a blood report PDF and extract BEI-relevant biomarkers.

    Returns PDFExtractionResult-compatible dict:
    {
        patient_id_detected: str | None,
        extracted_markers:   {canonical_marker: float},
        extraction_notes:    [str],
        raw_text_preview:    str,
        markers_found:       int,
        markers_missing:     [str],
        confidence:          str,
        unit_corrections:    [str],
    }
    """
    text = _extract_text_from_pdf_bytes(pdf_bytes)
    extracted: Dict[str, float] = {}
    corrections: List[str] = []
    notes: List[str] = []

    patient_id = _detect_patient_id(text)

    # Apply marker patterns
    for canonical, patterns, unit_hint in MARKER_PATTERNS:
        if canonical in extracted:
            continue  # already found from a higher-priority pattern
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                raw_val = _safe_float(m.group(1))
                if raw_val is None or raw_val < 0:
                    continue
                # Sanity range checks
                if canonical == "GCS" and not (3 <= raw_val <= 15):
                    notes.append(f"GCS value {raw_val} out of range [3-15] — skipped")
                    continue
                if canonical == "SOFA" and not (0 <= raw_val <= 24):
                    notes.append(f"SOFA value {raw_val} out of range [0-24] — skipped")
                    continue

                corrected_val, corr_note = _apply_unit_corrections(
                    canonical, raw_val, m.group(0)
                )
                if corr_note:
                    corrections.append(corr_note)

                extracted[canonical] = corrected_val
                logger.debug(
                    f"Extracted {canonical}={corrected_val} "
                    f"({unit_hint}) via pattern: {pat[:40]}"
                )
                break  # stop at first matching pattern for this marker

    # PaO2/FiO2 fallback from ABG
    if "PaO2_FiO2" not in extracted:
        result = _try_pao2_fio2_from_abg(text)
        if result:
            val, note = result
            extracted["PaO2_FiO2"] = val
            notes.append(f"Computed: {note}")

    # Confidence classification
    n_found = len(extracted)
    if n_found >= 12:
        confidence = "HIGH"
    elif n_found >= 7:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    missing = [m for m in ALL_BEI_MARKERS if m not in extracted]

    if confidence == "LOW":
        notes.append(
            f"LOW confidence extraction ({n_found}/17 markers found). "
            "Consider supplying cytokine panel values via marker_overrides "
            "or use POST /analyze with direct JSON input."
        )
    if "Lactate" not in extracted:
        notes.append(
            "Lactate not found in PDF. MNAR bias warning — BEI will be inflated. "
            "Lactate is mandatory for accurate BEI. Measure and supply via marker_overrides."
        )

    return {
        "patient_id_detected": patient_id,
        "extracted_markers":   extracted,
        "extraction_notes":    notes,
        "raw_text_preview":    text[:500],
        "markers_found":       n_found,
        "markers_missing":     missing,
        "confidence":          confidence,
        "unit_corrections":    corrections,
    }


def merge_markers(
    pdf_markers: Dict[str, float],
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Merge PDF-extracted markers with optional manual overrides.
    Overrides take precedence (allows user to fill cytokine panel gaps).
    Returns (merged_dict, list_of_applied_overrides).
    """
    merged = dict(pdf_markers)
    applied: List[str] = []
    if overrides:
        for marker, val in overrides.items():
            if val is not None:
                if marker in merged:
                    applied.append(
                        f"Override {marker}: {merged[marker]} → {val}"
                    )
                else:
                    applied.append(f"Supplied {marker}: {val}")
                merged[marker] = val
    return merged, applied
