"""
models.py — Pydantic request / response models for SepsisAI BEI API v2.0
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel


# ── Enumerations ──────────────────────────────────────────────────────────────

class BEIZone(str, Enum):
    VERY_LOW   = "VERY_LOW"
    AT_RISK    = "AT_RISK"
    SUFFICIENT = "SUFFICIENT"


class TippingTier(str, Enum):
    KILL     = "KILL"
    MODERATE = "MODERATE"
    NOISE    = "NOISE"
    BENIGN   = "BENIGN"
    UNKNOWN  = "UNKNOWN"


class BEIModule(str, Enum):
    A    = "A"
    B    = "B"
    BOTH = "both"


# ── Request models ────────────────────────────────────────────────────────────

class BiomarkerInput(BaseModel):
    """
    Direct structured input of biomarker values.

    All 17 BEI markers accepted; any subset ≥1 marker will compute.
    Missing markers reduce coverage_pct and may inflate BEI if Lactate
    is missing (MNAR bias — always measure Lactate).

    Units (MUST match Davoudian 2023 dataset units):
      D-dimer     : ng/mL
      SOFA        : score (0–24)
      Lactate     : mmol/L
      PTX3        : ng/mL
      Urea        : mmol/L
      IL-8        : pg/mL
      IL-10       : pg/mL
      GCS         : score (3–15)
      sIL-1R2     : ng/mL
      CRP         : mg/L
      PaO2_FiO2   : ratio (e.g. 200 = PaO2 200 mmHg / FiO2 1.0)
      IL-6        : pg/mL
      MAP         : mmHg
      Platelets   : ×10⁹/L
      IL-18       : pg/mL
      IL-1ra      : pg/mL
      TNF-alpha   : pg/mL
      Creatinine  : μmol/L
    """
    patient_id: str = Field(..., description="Hospital's own patient / admission ID")

    # Standard clinical labs
    Lactate:    Optional[float] = Field(None, ge=0, description="mmol/L")
    MAP:        Optional[float] = Field(None, ge=0, description="mmHg")
    GCS:        Optional[float] = Field(None, ge=3, le=15, description="3-15")
    Creatinine: Optional[float] = Field(None, ge=0, description="μmol/L")
    Urea:       Optional[float] = Field(None, ge=0, description="mmol/L")
    Platelets:  Optional[float] = Field(None, ge=0, description="×10⁹/L")
    CRP:        Optional[float] = Field(None, ge=0, description="mg/L")
    PaO2_FiO2: Optional[float] = Field(None, ge=0, description="ratio ×100 (e.g. 200)")
    SOFA:       Optional[float] = Field(None, ge=0, le=24, description="score 0-24")

    # Cytokine panel (Luminex/ELISA — separate from standard labs)
    PTX3:      Optional[float] = Field(None, ge=0, description="ng/mL")
    sIL_1R2:   Optional[float] = Field(None, ge=0, alias="sIL-1R2",  description="ng/mL")
    IL_10:     Optional[float] = Field(None, ge=0, alias="IL-10",    description="pg/mL")
    IL_8:      Optional[float] = Field(None, ge=0, alias="IL-8",     description="pg/mL")
    IL_6:      Optional[float] = Field(None, ge=0, alias="IL-6",     description="pg/mL")
    IL_18:     Optional[float] = Field(None, ge=0, alias="IL-18",    description="pg/mL")
    IL_1ra:    Optional[float] = Field(None, ge=0, alias="IL-1ra",   description="pg/mL")
    TNF_alpha: Optional[float] = Field(None, ge=0, alias="TNF-alpha",description="pg/mL")
    D_dimer:   Optional[float] = Field(None, ge=0, alias="D-dimer",  description="ng/mL")

    model_config = {"populate_by_name": True}

    def to_raw_dict(self) -> Dict[str, Optional[float]]:
        """Return marker dict keyed by canonical names used in BEI engine."""
        aliases = {
            "sIL_1R2":   "sIL-1R2",
            "IL_10":     "IL-10",
            "IL_8":      "IL-8",
            "IL_6":      "IL-6",
            "IL_18":     "IL-18",
            "IL_1ra":    "IL-1ra",
            "TNF_alpha": "TNF-alpha",
            "D_dimer":   "D-dimer",
        }
        out: Dict[str, Optional[float]] = {}
        for field_name, value in self.model_dump(exclude={"patient_id"}).items():
            canonical = aliases.get(field_name, field_name)
            out[canonical] = value
        return {k: v for k, v in out.items() if v is not None}


class AnalysisRequest(BaseModel):
    """Full analysis request: biomarkers → BEI + optional treatment synthesis."""
    biomarkers: BiomarkerInput
    include_treatment: bool = Field(
        True,
        description="If True, run RAG + Claude synthesis (requires ANTHROPIC_API_KEY)",
    )
    include_rag_chunks: bool = Field(
        False,
        description="If True, return raw retrieved literature chunks in response",
    )
    mrn: Optional[str] = Field(
        None,
        description="Medical record number, if different from patient_id",
    )
    module: BEIModule = Field(
        BEIModule.A,
        description=(
            "'A' = full 17-marker BEI (default, AUC=0.812); "
            "'B' = KILL-4 reduced panel (D-dimer, SOFA, Lactate, PTX3 — AUC unvalidated); "
            "'both' = run both engines and return a structured comparison block"
        ),
    )


# ── BEI response models ───────────────────────────────────────────────────────

class ModuleComparison(BaseModel):
    """Structured comparison of Module A (17-marker) vs Module B (KILL-4)."""
    zone_agreement:    bool
    score_delta:       float  = Field(..., description="Module A score minus Module B score")
    delta_note:        str
    tipping_agreement: bool
    module_a_score:    float
    module_b_score:    float
    module_a_zone:     str
    module_b_zone:     str
    module_a_tipping:  str
    module_b_tipping:  str
    module_a_coverage: float = 0.0
    module_b_coverage: float = 0.0
    validation_note:   str


class BEIResult(BaseModel):
    bei_score:             float     = Field(..., description="BEI composite [0=collapse, 1=homeostasis]")
    bei_zone:              BEIZone
    bei_zone_detail:       str
    energy_profile:        Dict[str, float]
    tipping_contributions: Dict[str, float]
    tipping_marker:        str
    tipping_tier:          TippingTier
    tipping_action:        str
    gcs_alarm:             bool      = Field(..., description="True if GCS ≤14 (67% 90d mortality)")
    gh5_score:             Optional[float] = Field(None, description="GH5 triage sub-score")
    missing_markers:       List[str]
    n_available:           int
    n_missing:             int
    coverage_pct:          float
    lactate_missing_warning: bool


class TreatmentSynthesis(BaseModel):
    patient_id:          str
    tipping_marker:      str
    tipping_tier:        str
    synthesis_text:      str       = Field(..., description="Full 6-section treatment protocol")
    rag_chunk_count:     int
    model_used:          str
    hard_rules_applied:  List[str]
    synthesized_at:      str


class RagChunk(BaseModel):
    text:   str
    score:  float
    source: str
    year:   Any


class AnalysisResponse(BaseModel):
    patient_id:          str
    analysis_id:         Optional[str] = Field(None, description="UUID of this analysis in the database")
    internal_patient_id: Optional[str] = Field(None, description="Internal patient UUID")
    module:              str           = Field("A", description="Module used: A, B, or both")
    bei:                 BEIResult
    module_b:            Optional[BEIResult]        = Field(None, description="KILL-4 result (module B or both)")
    comparison:          Optional[ModuleComparison] = Field(None, description="Module A vs B comparison (module=both)")
    treatment:           Optional[TreatmentSynthesis] = None
    rag_chunks:          Optional[List[RagChunk]]     = None
    unit_corrections:    List[str] = Field(default_factory=list)
    clinical_flags:      List[str]
    low_confidence_flag: bool
    low_confidence_note: str = ""
    analyzed_at:         str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    disclaimer:          str = (
        "INVESTIGATIONAL — Not FDA cleared. "
        "Decision-support only. Clinician makes final treatment decisions."
    )


# ── PDF extraction models ─────────────────────────────────────────────────────

class PDFExtractionResult(BaseModel):
    patient_id_detected: Optional[str] = None
    extracted_markers:   Dict[str, float] = Field(default_factory=dict)
    extraction_notes:    List[str]        = Field(default_factory=list)
    raw_text_preview:    str              = Field(default="")
    markers_found:       int              = 0
    markers_missing:     List[str]        = Field(default_factory=list)
    confidence:          str              = Field(
        default="LOW",
        description="HIGH (≥12/17 markers), MEDIUM (7-11), LOW (<7)",
    )


class PDFAnalysisRequest(BaseModel):
    patient_id:         str   = Field(..., description="Hospital patient / admission ID")
    include_treatment:  bool  = True
    include_rag_chunks: bool  = False
    marker_overrides:   Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Optional: supply markers the PDF parser missed, "
            "using canonical names (e.g. {'PTX3': 28.82, 'IL-10': 3.87})"
        ),
    )
    mrn: Optional[str] = None


# ── Hospital models ───────────────────────────────────────────────────────────

class HospitalCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255, description="Hospital legal name")
    tier: str = Field("standard", description="Tier: standard | premium | research")


class HospitalResponse(BaseModel):
    id:             str
    name:           str
    api_key_prefix: str   = Field(..., description="First 12 chars of API key — display only")
    tier:           str
    active:         bool
    created_at:     str

    model_config = {"from_attributes": True}


class HospitalRegistered(HospitalResponse):
    """Returned ONCE at registration — includes the raw API key."""
    api_key: str = Field(
        ...,
        description=(
            "Raw API key — store this immediately. "
            "It will NOT be shown again. "
            "Include as X-API-Key header on all subsequent requests."
        ),
    )


class HospitalStats(BaseModel):
    hospital_id:       str
    name:              str
    tier:              str
    total_patients:    int
    total_analyses:    int
    zone_distribution: Dict[str, int]


class KeyRotated(BaseModel):
    hospital_id:    str
    api_key:        str = Field(..., description="New raw API key — store immediately, shown ONCE")
    api_key_prefix: str


# ── Patient models ────────────────────────────────────────────────────────────

class PatientResponse(BaseModel):
    id:                  str
    external_patient_id: str
    mrn:                 Optional[str]
    analysis_count:      int = 0
    last_analyzed_at:    Optional[str] = None
    created_at:          str

    model_config = {"from_attributes": True}


# ── Analysis summary (list views) ─────────────────────────────────────────────

class AnalysisSummary(BaseModel):
    id:                  str
    patient_id:          str
    external_patient_id: str
    bei_score:           float
    bei_zone:            str
    tipping_marker:      Optional[str]
    tipping_tier:        Optional[str]
    coverage_pct:        float
    low_confidence:      bool
    input_source:        str
    analyzed_at:         str
    module:              str            = "A"
    bei_b_score:         Optional[float] = None
    bei_b_zone:          Optional[str]   = None

    model_config = {"from_attributes": True}


# ── Pagination ────────────────────────────────────────────────────────────────

T = TypeVar("T")


class PaginatedResponse(GenericModel, Generic[T]):
    items:   List[T]
    total:   int
    skip:    int
    limit:   int
    has_more: bool


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:            str
    database_ok:       bool
    chromadb_loaded:   bool
    chromadb_chunks:   Optional[int] = None
    anthropic_key_set: bool
    version:           str
