"""
db_models.py — SQLAlchemy ORM table models for SepsisAI hospital API
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Tables:
  hospitals          — one row per registered hospital / API client
  patients           — one row per (hospital, external_patient_id) pair
  analyses           — one BEI analysis result per admission / re-evaluation
  biomarker_readings — individual marker values + energy + TC per analysis
  clinical_flags     — alarm flags (GCS, lactate, U:Cr, etc.) per analysis
  unit_corrections   — unit conversion notes per analysis
  audit_logs         — immutable FDA audit trail
"""
from __future__ import annotations
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────

class Hospital(Base):
    """One row per hospital / API client."""
    __tablename__ = "hospitals"

    id             = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name           = Column(String(255), nullable=False, index=True)
    api_key_hash   = Column(String(64),  nullable=False, unique=True)   # SHA-256 hex
    api_key_prefix = Column(String(12),  nullable=False)                # display-safe prefix
    tier           = Column(String(50),  nullable=False, default="standard")
    active         = Column(Boolean,     nullable=False, default=True)
    created_at     = Column(DateTime,    nullable=False, default=_now)
    updated_at     = Column(DateTime,    nullable=False, default=_now, onupdate=_now)

    patients    = relationship("Patient",  back_populates="hospital", cascade="all, delete-orphan")
    analyses    = relationship("Analysis", back_populates="hospital")
    audit_logs  = relationship("AuditLog", back_populates="hospital")


class Patient(Base):
    """One row per (hospital, external_patient_id) pair — supports readmissions."""
    __tablename__ = "patients"
    __table_args__ = (
        UniqueConstraint("hospital_id", "external_patient_id", name="uq_hospital_patient"),
        Index("ix_patients_hospital_created", "hospital_id", "created_at"),
    )

    id                  = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    hospital_id         = Column(UUID(as_uuid=False), ForeignKey("hospitals.id"), nullable=False, index=True)
    external_patient_id = Column(String(255), nullable=False)   # hospital's own MRN / ID
    mrn                 = Column(String(255), nullable=True)     # medical record number if separate
    created_at          = Column(DateTime, nullable=False, default=_now)

    hospital  = relationship("Hospital", back_populates="patients")
    analyses  = relationship(
        "Analysis", back_populates="patient",
        order_by="Analysis.analyzed_at.desc()",
    )


class Analysis(Base):
    """
    One complete BEI analysis result — linked to a patient and hospital.
    All BEI fields are denormalized here for fast queries.
    Fine-grained biomarker data is in BiomarkerReading.
    """
    __tablename__ = "analyses"
    __table_args__ = (
        Index("ix_analyses_hospital_patient", "hospital_id", "patient_id"),
        Index("ix_analyses_hospital_zone",    "hospital_id", "bei_zone"),
        Index("ix_analyses_analyzed_at",      "analyzed_at"),
    )

    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    patient_id  = Column(UUID(as_uuid=False), ForeignKey("patients.id"),  nullable=False)
    hospital_id = Column(UUID(as_uuid=False), ForeignKey("hospitals.id"), nullable=False)

    # Denormalised for audit / reporting without joins
    external_patient_id = Column(String(255), nullable=False)

    # ── BEI core ──────────────────────────────────────────────────────────────
    bei_score       = Column(Float,       nullable=False)
    bei_zone        = Column(String(20),  nullable=False)
    bei_zone_detail = Column(Text,        nullable=True)
    tipping_marker  = Column(String(100), nullable=True)
    tipping_tier    = Column(String(20),  nullable=True)
    tipping_action  = Column(Text,        nullable=True)
    gh5_score       = Column(Float,       nullable=True)
    gcs_alarm       = Column(Boolean,     nullable=False, default=False)
    coverage_pct    = Column(Float,       nullable=False)
    n_available     = Column(Integer,     nullable=False)
    n_missing       = Column(Integer,     nullable=False)
    lactate_missing = Column(Boolean,     nullable=False, default=False)
    low_confidence  = Column(Boolean,     nullable=False, default=False)

    # ── Treatment synthesis ────────────────────────────────────────────────────
    synthesis_text     = Column(Text,        nullable=True)
    model_used         = Column(String(100), nullable=True)
    rag_chunk_count    = Column(Integer,     nullable=True)
    hard_rules_applied = Column(Text,        nullable=True)  # JSON array

    # ── Provenance ─────────────────────────────────────────────────────────────
    input_source  = Column(String(20),  nullable=False, default="json")   # 'json' | 'pdf'
    pdf_filename  = Column(String(255), nullable=True)
    analyzed_at   = Column(DateTime,    nullable=False, default=_now)
    created_at    = Column(DateTime,    nullable=False, default=_now)

    patient   = relationship("Patient",  back_populates="analyses")
    hospital  = relationship("Hospital", back_populates="analyses")

    biomarkers       = relationship("BiomarkerReading", back_populates="analysis", cascade="all, delete-orphan")
    clinical_flags   = relationship("ClinicalFlag",     back_populates="analysis", cascade="all, delete-orphan")
    unit_corrections = relationship("UnitCorrection",   back_populates="analysis", cascade="all, delete-orphan")


class BiomarkerReading(Base):
    """Raw value, energy score, and tipping contribution for one marker in one analysis."""
    __tablename__ = "biomarker_readings"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id = Column(UUID(as_uuid=False), ForeignKey("analyses.id"), nullable=False, index=True)
    marker_name = Column(String(100), nullable=False)
    raw_value   = Column(Float, nullable=True)
    energy_score= Column(Float, nullable=True)
    tc_value    = Column(Float, nullable=True)   # tipping contribution

    analysis = relationship("Analysis", back_populates="biomarkers")


class ClinicalFlag(Base):
    """One alarm / clinical flag string per analysis."""
    __tablename__ = "clinical_flags"

    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id = Column(UUID(as_uuid=False), ForeignKey("analyses.id"), nullable=False, index=True)
    flag_text   = Column(Text, nullable=False)

    analysis = relationship("Analysis", back_populates="clinical_flags")


class UnitCorrection(Base):
    """Unit conversion / extraction note applied to one analysis."""
    __tablename__ = "unit_corrections"

    id              = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    analysis_id     = Column(UUID(as_uuid=False), ForeignKey("analyses.id"), nullable=False, index=True)
    correction_text = Column(Text, nullable=False)

    analysis = relationship("Analysis", back_populates="unit_corrections")


class AuditLog(Base):
    """
    Immutable FDA audit trail — one row per significant API action.
    Never UPDATE or DELETE rows from this table.
    """
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_hospital_created", "hospital_id", "created_at"),
    )

    id          = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    hospital_id = Column(UUID(as_uuid=False), ForeignKey("hospitals.id"), nullable=True, index=True)
    action      = Column(String(100), nullable=False)    # 'analyze_json', 'analyze_pdf', ...
    analysis_id = Column(UUID(as_uuid=False), nullable=True)
    patient_id  = Column(UUID(as_uuid=False), nullable=True)
    ip_address  = Column(String(50),  nullable=True)
    status_code = Column(Integer,     nullable=True)
    error_msg   = Column(Text,        nullable=True)
    created_at  = Column(DateTime,    nullable=False, default=_now)

    hospital = relationship("Hospital", back_populates="audit_logs")
