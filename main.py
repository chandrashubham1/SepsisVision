"""
main.py — SepsisAI BEI API v2.0
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Hospital-facing production API:
  • Multi-tenant API key authentication (X-API-Key header)
  • PostgreSQL backend via SQLAlchemy
  • Patient longitudinal tracking
  • Full BEI computation + RAG-grounded treatment synthesis
  • Immutable FDA audit trail
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException,
    Request, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

import config
import crud
from auth import get_current_hospital, require_admin
from bei_engine import (
    BEI_WEIGHTS, BEI_ZONES, KILL4_WEIGHTS,
    build_clinical_flags, compare_modules, compute_bei, compute_bei_b,
)
from database import check_db_connection, create_tables, get_db
from db_models import Analysis, Hospital
from lab_parser import merge_markers, parse_blood_report_pdf
from models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisSummary,
    BEIModule,
    BEIResult,
    BEIZone,
    HealthResponse,
    HospitalCreate,
    HospitalRegistered,
    HospitalResponse,
    HospitalStats,
    KeyRotated,
    ModuleComparison,
    PaginatedResponse,
    PatientResponse,
    PDFExtractionResult,
    RagChunk,
    TippingTier,
    TreatmentSynthesis,
    XAIMarkerAttribution,
    XAIReport,
)
from rag_retrieval import RagRetriever
from treatment import synthesize_treatment
from xai_engine import build_xai_attribution, synthesize_rounds_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level RAG retriever (loaded once at startup) ───────────────────────
_retriever: Optional[RagRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever

    # Ensure database tables exist (idempotent — Alembic is preferred in prod)
    try:
        import asyncio as _asyncio
        await _asyncio.wait_for(
            _asyncio.get_event_loop().run_in_executor(None, create_tables),
            timeout=5.0,
        )
        logger.info("Database tables ready.")
    except Exception as exc:
        logger.warning("Database setup skipped (non-fatal — no DB connection?): %s", exc)

    # Load ChromaDB RAG retriever
    try:
        _retriever = RagRetriever(
            chroma_path=config.CHROMA_PATH,
            collection_name=config.COLLECTION_NAME,
            embed_model=config.EMBED_MODEL,
        )
        _retriever.load()
        logger.info("ChromaDB loaded: %d chunks", _retriever.chunk_count)
    except Exception as exc:
        logger.warning("ChromaDB unavailable at startup (non-fatal): %s", exc)
        _retriever = None

    yield

    logger.info("Shutting down SepsisAI API.")


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title=config.APP_TITLE,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Landing page ──────────────────────────────────────────────────────────────
import pathlib as _pathlib

_LANDING_HTML: str = (_pathlib.Path(__file__).parent / "landing.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """Serve the SepsisAI BEI API landing page."""
    return HTMLResponse(content=_LANDING_HTML)


# ── Analysis tool page ────────────────────────────────────────────────────────
_TOOL_HTML: str = (_pathlib.Path(__file__).parent / "tool.html").read_text(encoding="utf-8")


@app.get("/tool", response_class=HTMLResponse, include_in_schema=False)
def tool_page() -> HTMLResponse:
    """Serve the BEI Analysis Tool UI."""
    return HTMLResponse(content=_TOOL_HTML)


# Serve static assets (Sutra logo, etc.)
_static_dir = _pathlib.Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return (forwarded.split(",")[0].strip() if forwarded
            else request.client.host if request.client else "unknown")


def _make_bei_result(bei: dict) -> BEIResult:
    """Convert a raw BEI engine dict to a BEIResult Pydantic model."""
    return BEIResult(
        bei_score=bei["bei_score"],
        bei_zone=BEIZone(bei["bei_zone"]),
        bei_zone_detail=bei["bei_zone_detail"],
        energy_profile=bei["energy_profile"],
        tipping_contributions=bei["tipping_contributions"],
        tipping_marker=bei["tipping_marker"],
        tipping_tier=TippingTier(bei.get("tipping_tier", "UNKNOWN")),
        tipping_action=bei.get("tipping_action", ""),
        gcs_alarm=bei.get("gcs_alarm", False),
        gh5_score=bei.get("gh5_score"),
        missing_markers=bei.get("missing_markers", []),
        n_available=bei["n_available"],
        n_missing=bei["n_missing"],
        coverage_pct=bei["coverage_pct"],
        lactate_missing_warning=bei.get("lactate_missing_warning", False),
    )


def _build_analysis_response(
    *,
    patient_id:          str,
    analysis_id:         Optional[str],
    internal_patient_id: Optional[str],
    module:              str,
    bei:                 dict,
    bei_b:               Optional[dict],
    comparison:          Optional[dict],
    unit_corrections:    List[str],
    clinical_flags:      List[str],
    treatment_result:    Optional[dict],
    rag_chunks:          Optional[List[dict]],
    include_rag_chunks:  bool,
    xai_result:          Optional[dict] = None,
) -> AnalysisResponse:
    """Assemble an AnalysisResponse from raw BEI + optional Module B + treatment dicts."""
    bei_result   = _make_bei_result(bei)
    bei_b_result = _make_bei_result(bei_b) if bei_b else None

    comparison_obj: Optional[ModuleComparison] = None
    if comparison:
        comparison_obj = ModuleComparison(**comparison)

    treatment_obj: Optional[TreatmentSynthesis] = None
    if treatment_result:
        treatment_obj = TreatmentSynthesis(
            patient_id=patient_id,
            tipping_marker=bei["tipping_marker"],
            tipping_tier=bei.get("tipping_tier", "UNKNOWN"),
            synthesis_text=treatment_result["synthesis_text"],
            rag_chunk_count=treatment_result.get("rag_chunk_count", 0),
            model_used=treatment_result.get("model_used", config.CLAUDE_MODEL),
            hard_rules_applied=treatment_result.get("hard_rules_applied", []),
            synthesized_at=treatment_result.get("synthesized_at", ""),
        )

    chunks_out: Optional[List[RagChunk]] = None
    if include_rag_chunks and rag_chunks:
        chunks_out = [
            RagChunk(
                text=c.get("text", ""),
                score=c.get("score", 0.0),
                source=c.get("source", ""),
                year=c.get("year"),
            )
            for c in rag_chunks
        ]

    # ── xAI report ────────────────────────────────────────────────────────────
    xai_report_obj: Optional[XAIReport] = None
    xmd_document:   Optional[dict]      = None

    if xai_result:
        xai_data = xai_result.get("xai_attribution") or xai_result
        top_d    = xai_data.get("top_driver")
        xai_report_obj = XAIReport(
            module             = xai_data.get("module", module),
            bei_score          = xai_data.get("bei_score", 0.0),
            bei_zone           = xai_data.get("bei_zone", "UNKNOWN"),
            predicted_outcome  = xai_data.get("predicted_outcome", ""),
            zone_confidence    = xai_data.get("zone_confidence", ""),
            xai_summary        = xai_data.get("xai_summary", ""),
            n_markers          = xai_data.get("n_markers", 0),
            top_driver         = XAIMarkerAttribution(**top_d) if top_d else None,
            marker_attributions= [
                XAIMarkerAttribution(**a) for a in xai_data.get("marker_attributions", [])
            ],
            rounds_narrative   = xai_result.get("rounds_text"),
            rounds_model_used  = xai_result.get("model_used"),
        )

        # ── FDA xmd xAI document ─────────────────────────────────────────────
        # Build once per analysis; zero API cost (deterministic)
        try:
            from fda_xai_schema import build_xai_document, serialise_xai_document
            xmd_doc = build_xai_document(
                patient_id=patient_id,
                analysis_id=analysis_id or "",
                bei=bei,
                raw_input=xai_result.get("_raw_input", {}),
                xai_data=xai_data,
            )
            xmd_document = serialise_xai_document(xmd_doc)
        except Exception as exc:
            logger.warning("FDA xmd document generation failed (non-fatal): %s", exc)

    low_conf   = bei["coverage_pct"] < 60
    panel_size = bei.get("panel_size", 17)
    return AnalysisResponse(
        patient_id=patient_id,
        analysis_id=analysis_id,
        internal_patient_id=internal_patient_id,
        module=module,
        bei=bei_result,
        module_b=bei_b_result,
        comparison=comparison_obj,
        treatment=treatment_obj,
        xai_report=xai_report_obj,
        xmd_document=xmd_document,
        rag_chunks=chunks_out,
        unit_corrections=unit_corrections,
        clinical_flags=clinical_flags,
        low_confidence_flag=low_conf,
        low_confidence_note=(
            f"Only {bei['n_available']}/{panel_size} markers provided "
            f"(coverage {bei['coverage_pct']:.0f}%). "
            "BEI may be unreliable — obtain full panel before acting."
        ) if low_conf else "",
    )


def _run_analysis(
    *,
    request:            Request,
    hospital:           Hospital,
    external_patient_id: str,
    raw_input:          dict,
    include_treatment:  bool,
    include_rag_chunks: bool,
    include_xai:        bool = True,
    include_rounds:     bool = False,
    unit_corrections:   List[str],
    mrn:                Optional[str],
    db:                 Session,
    input_source:       str = "json",
    pdf_filename:       Optional[str] = None,
    module:             str = "A",
) -> AnalysisResponse:
    """
    Core analysis pipeline — called by all /analyze endpoints.
    1. Compute BEI (Module A, B, or both)
    2. Build clinical flags
    3. Optionally compute xAI attribution (deterministic — no API)
    4. Persist patient + analysis to database
    5. Optionally run RAG retrieval + treatment synthesis
    6. Optionally run doctor-on-rounds narrative (xAI → Claude API)
    7. Return AnalysisResponse
    """
    # ── 1. BEI computation ────────────────────────────────────────────────────
    bei:        dict              = compute_bei(raw_input)   # Module A always computed
    bei_b:      Optional[dict]   = None
    comparison: Optional[dict]   = None

    if module in ("B", "both"):
        bei_b = compute_bei_b(raw_input)

    if module == "both" and bei_b:
        comparison = compare_modules(bei, bei_b)

    # When module=B only, swap primary BEI to Module B for treatment synthesis context
    bei_primary = bei_b if module == "B" else bei

    # ── 2. Clinical flags ─────────────────────────────────────────────────────
    flags = build_clinical_flags(bei_primary, raw_input)

    # ── 3. xAI attribution (deterministic — no API call) ─────────────────────
    xai_result: Optional[dict] = None
    if include_xai:
        try:
            xai_data = build_xai_attribution(bei_primary, raw_input)
            xai_result = {
                "xai_attribution": xai_data,
                "_raw_input":      raw_input,  # passed to FDA xmd builder (not sent in API response)
            }
        except Exception as exc:
            logger.warning("xAI attribution failed (non-fatal): %s", exc)

    # ── 4. Optionally run treatment synthesis first (before DB write so we
    #        can store synthesis_text in one transaction)
    treatment_result: Optional[dict] = None
    rag_chunks:       Optional[List[dict]] = None

    if include_treatment:
        if not _retriever or not _retriever.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "RAG retriever not available. "
                    "ChromaDB collection may not be loaded. "
                    "BEI computation succeeded — retry with include_treatment=false "
                    "for BEI-only results."
                ),
            )
        if not config.ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ANTHROPIC_API_KEY not set — treatment synthesis unavailable.",
            )
        try:
            chunks = _retriever.retrieve_for_bei(
                bei_primary,
                top_k=config.TOP_K,
                final_k=config.FINAL_K,
            )
            rag_chunks = [
                {"text": c["text"], "score": c.get("score", 0.0),
                 "source": c.get("source", ""), "year": c.get("year")}
                for c in chunks
            ]
            treatment_result = synthesize_treatment(
                data={
                    "patient_id":      external_patient_id,
                    "bei":             bei_primary,
                    "raw_input":       raw_input,
                    # Pass xAI attribution so treatment prompt gets the
                    # per-marker TC table (deterministic, no extra API cost)
                    "xai_attribution": (
                        xai_result.get("xai_attribution") if xai_result else None
                    ),
                },
                chunks=chunks,
                api_key=config.ANTHROPIC_API_KEY,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Treatment synthesis failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Treatment synthesis failed: {exc}",
            )

    # ── 5. Doctor-on-rounds narrative (xAI → Claude API) ─────────────────────
    if include_rounds and include_xai and xai_result:
        if not config.ANTHROPIC_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ANTHROPIC_API_KEY not set — rounds narrative unavailable.",
            )
        try:
            rounds_result = synthesize_rounds_report(
                data={
                    "patient_id": external_patient_id,
                    "bei":        bei_primary,
                    "raw_input":  raw_input,
                    "flags":      flags,
                },
                api_key=config.ANTHROPIC_API_KEY,
            )
            # Merge rounds output into xai_result so _build_analysis_response sees it
            xai_result.update({
                "rounds_text": rounds_result.get("rounds_text"),
                "model_used":  rounds_result.get("model_used"),
            })
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Rounds narrative failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Rounds narrative synthesis failed: {exc}",
            )

    # ── 6. Persist to database (non-fatal when no DB is available) ────────────
    db_analysis_id:       Optional[str] = None
    internal_patient_id_: Optional[str] = None
    try:
        patient, _ = crud.get_or_create_patient(
            db=db,
            hospital_id=hospital.id,
            external_patient_id=external_patient_id,
            mrn=mrn,
        )
        db_analysis = crud.create_analysis(
            db=db,
            patient=patient,
            hospital_id=hospital.id,
            bei=bei_primary,
            raw_input=raw_input,
            unit_corrections=unit_corrections,
            clinical_flags=flags,
            treatment_result=treatment_result,
            input_source=input_source,
            pdf_filename=pdf_filename,
            module=module.upper(),
            bei_b=bei_b if module in ("B", "both") else None,
        )
        db_analysis_id       = db_analysis.id
        internal_patient_id_ = patient.id

        # ── 7. Audit log ───────────────────────────────────────────────────────
        crud.log_audit(
            db=db,
            action=f"analyze_{input_source}",
            hospital_id=hospital.id,
            analysis_id=db_analysis_id,
            patient_id=internal_patient_id_,
            ip_address=_get_client_ip(request),
            status_code=200,
        )
    except Exception as _db_exc:
        logger.warning(
            "Database persistence skipped (non-fatal — no DB connection?): %s", _db_exc
        )

    return _build_analysis_response(
        patient_id=external_patient_id,
        analysis_id=db_analysis_id,
        internal_patient_id=internal_patient_id_,
        module=module,
        bei=bei_primary,
        bei_b=bei_b if module == "both" else None,
        comparison=comparison,
        unit_corrections=unit_corrections,
        clinical_flags=flags,
        treatment_result=treatment_result,
        rag_chunks=rag_chunks,
        include_rag_chunks=include_rag_chunks,
        xai_result=xai_result,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Reference / health (no auth) ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Reference"])
def health(db: Session = Depends(get_db)):
    """System health — no authentication required."""
    return HealthResponse(
        status="ok",
        database_ok=check_db_connection(),
        chromadb_loaded=_retriever.is_loaded if _retriever else False,
        chromadb_chunks=_retriever.chunk_count if (_retriever and _retriever.is_loaded) else None,
        anthropic_key_set=bool(config.ANTHROPIC_API_KEY),
        version=config.APP_VERSION,
    )


@app.get("/bei/zones", tags=["Reference"])
def bei_zones():
    """BEI zone definitions with mortality data."""
    return {
        zone: {
            "range": f"[{lo}, {hi})",
            "mortality_90d": detail,
        }
        for zone, (lo, hi, detail) in BEI_ZONES.items()
    }


@app.get("/bei/markers", tags=["Reference"])
def bei_markers():
    """All 17 BEI markers with weights and descriptions."""
    return {
        marker: {
            "weight": weight,
            "rank": rank + 1,
        }
        for rank, (marker, weight) in enumerate(
            sorted(BEI_WEIGHTS.items(), key=lambda x: -x[1])
        )
    }


@app.post("/extract/pdf", response_model=PDFExtractionResult, tags=["Reference"])
async def extract_pdf(file: UploadFile = File(...)):
    """
    Parse a blood-report PDF and return extracted markers — without running BEI.
    Use to preview what the PDF parser found before submitting for analysis.
    No authentication required.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF.",
        )
    pdf_bytes = await file.read()
    result = parse_blood_report_pdf(pdf_bytes)
    return PDFExtractionResult(
        patient_id_detected=result.get("patient_id_detected"),
        extracted_markers=result.get("markers", {}),
        extraction_notes=result.get("notes", []),
        raw_text_preview=result.get("raw_text_preview", ""),
        markers_found=len(result.get("markers", {})),
        markers_missing=result.get("missing_markers", []),
        confidence=result.get("confidence", "LOW"),
    )


# ── Admin: hospital management ─────────────────────────────────────────────────

@app.post(
    "/admin/hospitals",
    response_model=HospitalRegistered,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)
def register_hospital(
    body: HospitalCreate,
    db:   Session = Depends(get_db),
):
    """
    Register a new hospital and issue an API key.
    The raw API key is returned ONCE — store it immediately.
    Requires ADMIN_API_KEY in X-API-Key header.
    """
    hospital, raw_key = crud.create_hospital(db=db, name=body.name, tier=body.tier)
    return HospitalRegistered(
        id=hospital.id,
        name=hospital.name,
        api_key_prefix=hospital.api_key_prefix,
        tier=hospital.tier,
        active=hospital.active,
        created_at=hospital.created_at.isoformat(),
        api_key=raw_key,
    )


@app.get(
    "/admin/hospitals",
    response_model=PaginatedResponse[HospitalResponse],
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)
def list_all_hospitals(
    skip:  int = 0,
    limit: int = 50,
    db:    Session = Depends(get_db),
):
    """List all registered hospitals. Requires admin key."""
    hospitals, total = crud.list_hospitals(db=db, skip=skip, limit=limit)
    return PaginatedResponse(
        items=[
            HospitalResponse(
                id=h.id,
                name=h.name,
                api_key_prefix=h.api_key_prefix,
                tier=h.tier,
                active=h.active,
                created_at=h.created_at.isoformat(),
            )
            for h in hospitals
        ],
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total,
    )


@app.patch(
    "/admin/hospitals/{hospital_id}/deactivate",
    response_model=HospitalResponse,
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)
def deactivate_hospital(
    hospital_id: str,
    db:          Session = Depends(get_db),
):
    """Deactivate a hospital (revokes all API key access). Requires admin key."""
    h = crud.set_hospital_active(db=db, hospital_id=hospital_id, active=False)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    return HospitalResponse(
        id=h.id, name=h.name, api_key_prefix=h.api_key_prefix,
        tier=h.tier, active=h.active, created_at=h.created_at.isoformat(),
    )


@app.patch(
    "/admin/hospitals/{hospital_id}/activate",
    response_model=HospitalResponse,
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)
def activate_hospital(
    hospital_id: str,
    db:          Session = Depends(get_db),
):
    """Re-activate a deactivated hospital. Requires admin key."""
    h = crud.set_hospital_active(db=db, hospital_id=hospital_id, active=True)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    return HospitalResponse(
        id=h.id, name=h.name, api_key_prefix=h.api_key_prefix,
        tier=h.tier, active=h.active, created_at=h.created_at.isoformat(),
    )


@app.post(
    "/admin/hospitals/{hospital_id}/rotate-key",
    response_model=KeyRotated,
    tags=["Admin"],
    dependencies=[Depends(require_admin)],
)
def rotate_hospital_key(
    hospital_id: str,
    db:          Session = Depends(get_db),
):
    """
    Issue a new API key for a hospital, immediately invalidating the old one.
    The new raw key is returned ONCE.
    """
    h = crud.get_hospital(db=db, hospital_id=hospital_id)
    if not h:
        raise HTTPException(status_code=404, detail="Hospital not found.")
    h, raw_key = crud.rotate_api_key(db=db, hospital=h)
    return KeyRotated(
        hospital_id=h.id,
        api_key=raw_key,
        api_key_prefix=h.api_key_prefix,
    )


# ── Hospital self-service ──────────────────────────────────────────────────────

@app.get("/hospitals/me", response_model=HospitalStats, tags=["Hospital"])
def get_my_hospital(
    hospital: Hospital = Depends(get_current_hospital),
    db:       Session  = Depends(get_db),
):
    """Returns the authenticated hospital's profile and usage statistics."""
    stats = crud.hospital_stats(db=db, hospital_id=hospital.id)
    return HospitalStats(
        hospital_id=hospital.id,
        name=hospital.name,
        tier=hospital.tier,
        **stats,
    )


# ── Patient management ─────────────────────────────────────────────────────────

@app.get(
    "/patients",
    response_model=PaginatedResponse[PatientResponse],
    tags=["Patients"],
)
def list_patients(
    skip:     int = 0,
    limit:    int = 50,
    hospital: Hospital = Depends(get_current_hospital),
    db:       Session  = Depends(get_db),
):
    """List all patients for the authenticated hospital."""
    patients, total = crud.list_patients(db=db, hospital_id=hospital.id, skip=skip, limit=limit)

    def _to_response(p) -> PatientResponse:
        analyses = p.analyses  # pre-loaded via relationship
        return PatientResponse(
            id=p.id,
            external_patient_id=p.external_patient_id,
            mrn=p.mrn,
            analysis_count=len(analyses),
            last_analyzed_at=analyses[0].analyzed_at.isoformat() if analyses else None,
            created_at=p.created_at.isoformat(),
        )

    return PaginatedResponse(
        items=[_to_response(p) for p in patients],
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total,
    )


@app.get(
    "/patients/{patient_id}",
    response_model=PatientResponse,
    tags=["Patients"],
)
def get_patient(
    patient_id: str,
    hospital:   Hospital = Depends(get_current_hospital),
    db:         Session  = Depends(get_db),
):
    """Get a patient by internal UUID (scoped to the calling hospital)."""
    p = crud.get_patient(db=db, hospital_id=hospital.id, patient_id=patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    analyses = p.analyses
    return PatientResponse(
        id=p.id,
        external_patient_id=p.external_patient_id,
        mrn=p.mrn,
        analysis_count=len(analyses),
        last_analyzed_at=analyses[0].analyzed_at.isoformat() if analyses else None,
        created_at=p.created_at.isoformat(),
    )


@app.get(
    "/patients/{patient_id}/analyses",
    response_model=PaginatedResponse[AnalysisSummary],
    tags=["Patients"],
)
def get_patient_analyses(
    patient_id: str,
    skip:  int = 0,
    limit: int = 20,
    hospital: Hospital = Depends(get_current_hospital),
    db:       Session  = Depends(get_db),
):
    """Full analysis history for a patient (newest first)."""
    p = crud.get_patient(db=db, hospital_id=hospital.id, patient_id=patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found.")
    analyses, total = crud.list_patient_analyses(
        db=db, hospital_id=hospital.id, patient_id=patient_id, skip=skip, limit=limit,
    )
    return PaginatedResponse(
        items=[
            AnalysisSummary(
                id=a.id,
                patient_id=a.patient_id,
                external_patient_id=a.external_patient_id,
                bei_score=a.bei_score,
                bei_zone=a.bei_zone,
                tipping_marker=a.tipping_marker,
                tipping_tier=a.tipping_tier,
                coverage_pct=a.coverage_pct,
                low_confidence=a.low_confidence,
                input_source=a.input_source,
                analyzed_at=a.analyzed_at.isoformat(),
                module=getattr(a, "module", "A") or "A",
                bei_b_score=getattr(a, "bei_b_score", None),
                bei_b_zone=getattr(a, "bei_b_zone", None),
            )
            for a in analyses
        ],
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + limit < total,
    )


# ── Analysis: JSON ─────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
)
def analyze_json(
    body:     AnalysisRequest,
    request:  Request,
    hospital: Hospital = Depends(get_current_hospital),
    db:       Session  = Depends(get_db),
):
    """
    Submit structured biomarker values for BEI analysis.

    Supply the patient's biomarkers in the request body.
    Set `include_treatment=true` (default) to receive a RAG-grounded
    treatment synthesis from Claude Sonnet.
    `patient_id` is your hospital's own identifier — the API will track
    multiple analyses for the same patient across admissions.
    """
    raw_input = body.biomarkers.to_raw_dict()
    if not raw_input:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one biomarker value is required.",
        )
    return _run_analysis(
        request=request,
        hospital=hospital,
        external_patient_id=body.biomarkers.patient_id,
        raw_input=raw_input,
        include_treatment=body.include_treatment,
        include_rag_chunks=body.include_rag_chunks,
        include_xai=body.include_xai,
        include_rounds=body.include_rounds,
        unit_corrections=[],
        mrn=body.mrn,
        db=db,
        input_source="json",
        module=body.module.value,
    )


# ── Analysis: PDF ──────────────────────────────────────────────────────────────

@app.post(
    "/analyze/pdf",
    response_model=AnalysisResponse,
    tags=["Analysis"],
)
async def analyze_pdf(
    request:           Request,
    file:              UploadFile = File(..., description="Blood-report PDF"),
    patient_id:        str        = Form(..., description="Your hospital's patient / admission ID"),
    include_treatment: bool       = Form(True),
    include_rag_chunks: bool      = Form(False),
    include_xai:       bool       = Form(True,  description="Include xAI attribution block (deterministic — no API call)"),
    include_rounds:    bool       = Form(False, description="Include doctor-on-rounds narrative (Claude API call)"),
    mrn:               Optional[str] = Form(None),
    module:            str           = Form(
        "A",
        description="'A' = full 17-marker BEI (default); 'B' = KILL-4 reduced panel; 'both' = run both",
    ),
    marker_overrides:  Optional[str] = Form(
        None,
        description='JSON object with additional marker values, e.g. \'{"PTX3": 28.82, "IL-10": 3.87}\'',
    ),
    hospital: Hospital = Depends(get_current_hospital),
    db:       Session  = Depends(get_db),
):
    """
    Upload a blood-report PDF for automated parsing + BEI analysis.

    The PDF parser handles standard hospital labs (CBC, CMP, ABG, coagulation).
    Cytokine markers (PTX3, sIL-1R2, IL-10, IL-8, IL-18, IL-1ra, TNF-alpha)
    are rarely in standard PDFs — supply them via `marker_overrides` JSON if
    available from a Luminex/ELISA report.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF.",
        )

    pdf_bytes = await file.read()
    parse_result = parse_blood_report_pdf(pdf_bytes)
    pdf_markers  = parse_result.get("markers", {})
    corrections  = parse_result.get("corrections", [])

    # Merge cytokine overrides (supply panel markers the PDF never contains)
    overrides: dict = {}
    if marker_overrides:
        try:
            overrides = json.loads(marker_overrides)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="marker_overrides must be a valid JSON object.",
            )

    raw_input = merge_markers(pdf_markers, overrides)
    if not raw_input:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No biomarker values could be extracted from the PDF "
                "and no marker_overrides were supplied."
            ),
        )

    # Validate module value from Form field
    module_clean = module.strip() if module else "A"
    if module_clean not in ("A", "B", "both"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="module must be one of: A, B, both",
        )

    return _run_analysis(
        request=request,
        hospital=hospital,
        external_patient_id=patient_id,
        raw_input=raw_input,
        include_treatment=include_treatment,
        include_rag_chunks=include_rag_chunks,
        include_xai=include_xai,
        include_rounds=include_rounds,
        unit_corrections=corrections,
        mrn=mrn,
        db=db,
        input_source="pdf",
        pdf_filename=file.filename,
        module=module_clean,
    )


# ── Analysis: retrieve ─────────────────────────────────────────────────────────

@app.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    tags=["Analysis"],
)
def get_analysis(
    analysis_id: str,
    hospital:    Hospital = Depends(get_current_hospital),
    db:          Session  = Depends(get_db),
):
    """Retrieve a stored analysis by its UUID (scoped to your hospital)."""
    a: Optional[Analysis] = crud.get_analysis(
        db=db, analysis_id=analysis_id, hospital_id=hospital.id,
    )
    if not a:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    # Reconstruct BEI dict from ORM fields (for response assembly)
    energy_profile = {r.marker_name: r.energy_score for r in a.biomarkers if r.energy_score is not None}
    tc_dict        = {r.marker_name: r.tc_value      for r in a.biomarkers if r.tc_value      is not None}
    stored_module  = getattr(a, "module", "A") or "A"

    bei = {
        "bei_score":               a.bei_score,
        "bei_zone":                a.bei_zone,
        "bei_zone_detail":         a.bei_zone_detail or "",
        "energy_profile":          energy_profile,
        "tipping_contributions":   tc_dict,
        "tipping_marker":          a.tipping_marker or "",
        "tipping_tier":            a.tipping_tier or "UNKNOWN",
        "tipping_action":          a.tipping_action or "",
        "gcs_alarm":               a.gcs_alarm,
        "gh5_score":               a.gh5_score,
        "missing_markers":         [],   # not stored separately; use coverage
        "n_available":             a.n_available,
        "n_missing":               a.n_missing,
        "coverage_pct":            a.coverage_pct,
        "lactate_missing_warning": a.lactate_missing,
        "panel_size":              len(KILL4_WEIGHTS) if stored_module == "B" else 17,
    }

    # Reconstruct a minimal Module B dict from stored columns (if present)
    bei_b_reconstructed: Optional[dict] = None
    if stored_module in ("B", "BOTH") and getattr(a, "bei_b_score", None) is not None:
        bei_b_reconstructed = {
            "bei_score":               a.bei_b_score,
            "bei_zone":                a.bei_b_zone or "UNKNOWN",
            "bei_zone_detail":         "",
            "energy_profile":          {},
            "tipping_contributions":   {},
            "tipping_marker":          "",
            "tipping_tier":            "UNKNOWN",
            "tipping_action":          "",
            "gcs_alarm":               False,
            "gh5_score":               None,
            "missing_markers":         [],
            "n_available":             0,
            "n_missing":               len(KILL4_WEIGHTS),
            "coverage_pct":            0.0,
            "lactate_missing_warning": False,
            "panel_size":              len(KILL4_WEIGHTS),
        }

    treatment_result = None
    if a.synthesis_text:
        treatment_result = {
            "synthesis_text":    a.synthesis_text,
            "model_used":        a.model_used or "",
            "rag_chunk_count":   a.rag_chunk_count or 0,
            "hard_rules_applied": json.loads(a.hard_rules_applied) if a.hard_rules_applied else [],
            "synthesized_at":    a.analyzed_at.isoformat(),
        }

    return _build_analysis_response(
        patient_id=a.external_patient_id,
        analysis_id=a.id,
        internal_patient_id=a.patient_id,
        module=stored_module,
        bei=bei,
        bei_b=bei_b_reconstructed if stored_module == "BOTH" else None,
        comparison=None,   # comparison not persisted; recomputable from scores if needed
        unit_corrections=[c.correction_text for c in a.unit_corrections],
        clinical_flags=[f.flag_text for f in a.clinical_flags],
        treatment_result=treatment_result,
        rag_chunks=None,
        include_rag_chunks=False,
    )
