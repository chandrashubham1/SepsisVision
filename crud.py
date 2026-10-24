"""
crud.py — Database CRUD operations for SepsisAI hospital API
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from auth import generate_api_key, api_key_prefix
from db_models import (
    Analysis, AuditLog, BiomarkerReading, ClinicalFlag,
    Hospital, Patient, UnitCorrection,
)

logger = logging.getLogger(__name__)


# ── Hospital ──────────────────────────────────────────────────────────────────

def create_hospital(
    db:   Session,
    name: str,
    tier: str = "standard",
) -> Tuple[Hospital, str]:
    """
    Register a new hospital and generate its API key.
    Returns (Hospital ORM object, raw_api_key).
    raw_api_key is shown ONCE — only the hash is persisted.
    """
    raw_key, key_hash = generate_api_key()
    hospital = Hospital(
        name=name,
        api_key_hash=key_hash,
        api_key_prefix=api_key_prefix(raw_key),
        tier=tier,
        active=True,
    )
    db.add(hospital)
    db.commit()
    db.refresh(hospital)
    logger.info("Hospital registered: name=%s id=%s", name, hospital.id)
    return hospital, raw_key


def rotate_api_key(db: Session, hospital: Hospital) -> Tuple[Hospital, str]:
    """
    Issue a new API key for a hospital, invalidating the old one.
    Returns (hospital, new_raw_key).
    """
    raw_key, key_hash = generate_api_key()
    hospital.api_key_hash   = key_hash
    hospital.api_key_prefix = api_key_prefix(raw_key)
    db.commit()
    db.refresh(hospital)
    logger.info("API key rotated for hospital id=%s", hospital.id)
    return hospital, raw_key


def get_hospital(db: Session, hospital_id: str) -> Optional[Hospital]:
    return db.query(Hospital).filter(Hospital.id == hospital_id).first()


def list_hospitals(
    db:    Session,
    skip:  int = 0,
    limit: int = 100,
) -> Tuple[List[Hospital], int]:
    q     = db.query(Hospital)
    total = q.count()
    return q.order_by(Hospital.created_at.desc()).offset(skip).limit(limit).all(), total


def set_hospital_active(
    db:          Session,
    hospital_id: str,
    active:      bool,
) -> Optional[Hospital]:
    h = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if h:
        h.active = active
        db.commit()
        db.refresh(h)
    return h


def hospital_stats(db: Session, hospital_id: str) -> dict:
    total_patients = (
        db.query(func.count(Patient.id))
        .filter(Patient.hospital_id == hospital_id)
        .scalar() or 0
    )
    total_analyses = (
        db.query(func.count(Analysis.id))
        .filter(Analysis.hospital_id == hospital_id)
        .scalar() or 0
    )
    zones = (
        db.query(Analysis.bei_zone, func.count(Analysis.id))
        .filter(Analysis.hospital_id == hospital_id)
        .group_by(Analysis.bei_zone)
        .all()
    )
    return {
        "total_patients":    total_patients,
        "total_analyses":    total_analyses,
        "zone_distribution": {z: c for z, c in zones},
    }


# ── Patient ───────────────────────────────────────────────────────────────────

def get_or_create_patient(
    db:                  Session,
    hospital_id:         str,
    external_patient_id: str,
    mrn:                 Optional[str] = None,
) -> Tuple[Patient, bool]:
    """
    Look up a patient by (hospital_id, external_patient_id).
    Creates and returns a new record if not found.
    Returns (Patient, created: bool).
    Supports readmissions — each new /analyze call creates a new Analysis row
    linked to the same Patient row.
    """
    patient = (
        db.query(Patient)
        .filter(
            Patient.hospital_id         == hospital_id,
            Patient.external_patient_id == external_patient_id,
        )
        .first()
    )
    if patient:
        return patient, False

    patient = Patient(
        hospital_id=hospital_id,
        external_patient_id=external_patient_id,
        mrn=mrn,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(
        "New patient created: ext_id=%s hospital=%s",
        external_patient_id, hospital_id,
    )
    return patient, True


def get_patient(
    db:          Session,
    hospital_id: str,
    patient_id:  str,
) -> Optional[Patient]:
    """Get patient by internal UUID, scoped to the calling hospital."""
    return (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.hospital_id == hospital_id)
        .first()
    )


def get_patient_by_external_id(
    db:                  Session,
    hospital_id:         str,
    external_patient_id: str,
) -> Optional[Patient]:
    return (
        db.query(Patient)
        .filter(
            Patient.hospital_id         == hospital_id,
            Patient.external_patient_id == external_patient_id,
        )
        .first()
    )


def list_patients(
    db:          Session,
    hospital_id: str,
    skip:        int = 0,
    limit:       int = 50,
) -> Tuple[List[Patient], int]:
    q     = db.query(Patient).filter(Patient.hospital_id == hospital_id)
    total = q.count()
    rows  = q.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()
    return rows, total


# ── Analysis ──────────────────────────────────────────────────────────────────

def create_analysis(
    db:               Session,
    patient:          Patient,
    hospital_id:      str,
    bei:              dict,
    raw_input:        dict,
    unit_corrections: List[str],
    clinical_flags:   List[str],
    treatment_result: Optional[dict],
    input_source:     str = "json",
    pdf_filename:     Optional[str] = None,
) -> Analysis:
    """
    Persist a complete BEI analysis to the database.
    Child rows (BiomarkerReading, ClinicalFlag, UnitCorrection) are
    written in the same transaction.
    """
    analysis = Analysis(
        patient_id=patient.id,
        hospital_id=hospital_id,
        external_patient_id=patient.external_patient_id,

        bei_score=bei["bei_score"],
        bei_zone=bei["bei_zone"],
        bei_zone_detail=bei.get("bei_zone_detail", ""),
        tipping_marker=bei.get("tipping_marker"),
        tipping_tier=bei.get("tipping_tier"),
        tipping_action=bei.get("tipping_action"),
        gh5_score=bei.get("gh5_score"),
        gcs_alarm=bool(bei.get("gcs_alarm", False)),
        coverage_pct=float(bei.get("coverage_pct", 0.0)),
        n_available=int(bei.get("n_available", 0)),
        n_missing=int(bei.get("n_missing", 0)),
        lactate_missing=bool(bei.get("lactate_missing_warning", False)),
        low_confidence=bei.get("coverage_pct", 100) < 60,

        synthesis_text=(
            treatment_result.get("synthesis_text") if treatment_result else None
        ),
        model_used=(
            treatment_result.get("model_used") if treatment_result else None
        ),
        rag_chunk_count=(
            treatment_result.get("rag_chunk_count") if treatment_result else None
        ),
        hard_rules_applied=(
            json.dumps(treatment_result.get("hard_rules_applied", []))
            if treatment_result else None
        ),

        input_source=input_source,
        pdf_filename=pdf_filename,
        analyzed_at=datetime.utcnow(),
    )
    db.add(analysis)
    db.flush()  # obtain analysis.id before inserting child rows

    # Per-marker readings
    energy_profile = bei.get("energy_profile", {})
    tc             = bei.get("tipping_contributions", {})
    for marker, value in raw_input.items():
        db.add(BiomarkerReading(
            analysis_id=analysis.id,
            marker_name=marker,
            raw_value=float(value) if value is not None else None,
            energy_score=energy_profile.get(marker),
            tc_value=tc.get(marker),
        ))

    for flag in clinical_flags:
        db.add(ClinicalFlag(analysis_id=analysis.id, flag_text=flag))

    for correction in unit_corrections:
        db.add(UnitCorrection(analysis_id=analysis.id, correction_text=correction))

    db.commit()
    db.refresh(analysis)
    logger.info(
        "Analysis stored: id=%s patient=%s BEI=%.4f [%s]",
        analysis.id, patient.external_patient_id,
        bei["bei_score"], bei["bei_zone"],
    )
    return analysis


def get_analysis(
    db:          Session,
    analysis_id: str,
    hospital_id: str,
) -> Optional[Analysis]:
    """Retrieve an analysis by UUID, scoped to the calling hospital."""
    return (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id, Analysis.hospital_id == hospital_id)
        .first()
    )


def list_patient_analyses(
    db:          Session,
    hospital_id: str,
    patient_id:  str,
    skip:        int = 0,
    limit:       int = 20,
) -> Tuple[List[Analysis], int]:
    q = db.query(Analysis).filter(
        Analysis.hospital_id == hospital_id,
        Analysis.patient_id  == patient_id,
    )
    total = q.count()
    rows  = q.order_by(Analysis.analyzed_at.desc()).offset(skip).limit(limit).all()
    return rows, total


def list_hospital_analyses(
    db:          Session,
    hospital_id: str,
    skip:        int = 0,
    limit:       int = 50,
) -> Tuple[List[Analysis], int]:
    q     = db.query(Analysis).filter(Analysis.hospital_id == hospital_id)
    total = q.count()
    rows  = q.order_by(Analysis.analyzed_at.desc()).offset(skip).limit(limit).all()
    return rows, total


# ── Audit log ─────────────────────────────────────────────────────────────────

def log_audit(
    db:          Session,
    action:      str,
    hospital_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    patient_id:  Optional[str] = None,
    ip_address:  Optional[str] = None,
    status_code: int = 200,
    error_msg:   Optional[str] = None,
) -> None:
    """
    Write an immutable audit log entry.
    Swallows exceptions so audit failures never disrupt the main request.
    """
    try:
        db.add(AuditLog(
            hospital_id=hospital_id,
            action=action,
            analysis_id=analysis_id,
            patient_id=patient_id,
            ip_address=ip_address,
            status_code=status_code,
            error_msg=error_msg,
        ))
        db.commit()
    except Exception as exc:
        logger.warning("Audit log write failed (non-fatal): %s", exc)
