"""Initial schema — all 7 SepsisAI tables

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── hospitals ──────────────────────────────────────────────────────────────
    op.create_table(
        "hospitals",
        sa.Column("id",             postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("name",           sa.String(255), nullable=False),
        sa.Column("api_key_hash",   sa.String(64),  nullable=False, unique=True),
        sa.Column("api_key_prefix", sa.String(12),  nullable=False),
        sa.Column("tier",           sa.String(50),  nullable=False, server_default="standard"),
        sa.Column("active",         sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("created_at",     sa.DateTime(),  nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at",     sa.DateTime(),  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_hospitals_name", "hospitals", ["name"])

    # ── patients ───────────────────────────────────────────────────────────────
    op.create_table(
        "patients",
        sa.Column("id",                  postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("hospital_id",         postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("hospitals.id"), nullable=False),
        sa.Column("external_patient_id", sa.String(255), nullable=False),
        sa.Column("mrn",                 sa.String(255), nullable=True),
        sa.Column("created_at",          sa.DateTime(), nullable=False, server_default=sa.text("now()")),
    )
    op.create_unique_constraint(
        "uq_hospital_patient", "patients", ["hospital_id", "external_patient_id"],
    )
    op.create_index("ix_patients_hospital_id",      "patients", ["hospital_id"])
    op.create_index("ix_patients_hospital_created", "patients", ["hospital_id", "created_at"])

    # ── analyses ───────────────────────────────────────────────────────────────
    op.create_table(
        "analyses",
        sa.Column("id",          postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("patient_id",  postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("patients.id"),  nullable=False),
        sa.Column("hospital_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("hospitals.id"), nullable=False),
        sa.Column("external_patient_id", sa.String(255), nullable=False),

        # BEI core
        sa.Column("bei_score",       sa.Float(),       nullable=False),
        sa.Column("bei_zone",        sa.String(20),    nullable=False),
        sa.Column("bei_zone_detail", sa.Text(),        nullable=True),
        sa.Column("tipping_marker",  sa.String(100),   nullable=True),
        sa.Column("tipping_tier",    sa.String(20),    nullable=True),
        sa.Column("tipping_action",  sa.Text(),        nullable=True),
        sa.Column("gh5_score",       sa.Float(),       nullable=True),
        sa.Column("gcs_alarm",       sa.Boolean(),     nullable=False, server_default="false"),
        sa.Column("coverage_pct",    sa.Float(),       nullable=False),
        sa.Column("n_available",     sa.Integer(),     nullable=False),
        sa.Column("n_missing",       sa.Integer(),     nullable=False),
        sa.Column("lactate_missing", sa.Boolean(),     nullable=False, server_default="false"),
        sa.Column("low_confidence",  sa.Boolean(),     nullable=False, server_default="false"),

        # Treatment
        sa.Column("synthesis_text",     sa.Text(),        nullable=True),
        sa.Column("model_used",         sa.String(100),   nullable=True),
        sa.Column("rag_chunk_count",    sa.Integer(),     nullable=True),
        sa.Column("hard_rules_applied", sa.Text(),        nullable=True),

        # Provenance
        sa.Column("input_source",  sa.String(20),  nullable=False, server_default="json"),
        sa.Column("pdf_filename",  sa.String(255), nullable=True),
        sa.Column("analyzed_at",   sa.DateTime(),  nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at",    sa.DateTime(),  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_analyses_hospital_patient", "analyses", ["hospital_id", "patient_id"])
    op.create_index("ix_analyses_hospital_zone",    "analyses", ["hospital_id", "bei_zone"])
    op.create_index("ix_analyses_analyzed_at",      "analyses", ["analyzed_at"])

    # ── biomarker_readings ─────────────────────────────────────────────────────
    op.create_table(
        "biomarker_readings",
        sa.Column("id",           postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("analysis_id",  postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("marker_name",  sa.String(100), nullable=False),
        sa.Column("raw_value",    sa.Float(),     nullable=True),
        sa.Column("energy_score", sa.Float(),     nullable=True),
        sa.Column("tc_value",     sa.Float(),     nullable=True),
    )
    op.create_index("ix_biomarker_readings_analysis_id", "biomarker_readings", ["analysis_id"])

    # ── clinical_flags ─────────────────────────────────────────────────────────
    op.create_table(
        "clinical_flags",
        sa.Column("id",          postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("flag_text",   sa.Text(), nullable=False),
    )
    op.create_index("ix_clinical_flags_analysis_id", "clinical_flags", ["analysis_id"])

    # ── unit_corrections ───────────────────────────────────────────────────────
    op.create_table(
        "unit_corrections",
        sa.Column("id",              postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("analysis_id",     postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("correction_text", sa.Text(), nullable=False),
    )
    op.create_index("ix_unit_corrections_analysis_id", "unit_corrections", ["analysis_id"])

    # ── audit_logs ─────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id",          postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("hospital_id", postgresql.UUID(as_uuid=False),
                  sa.ForeignKey("hospitals.id"), nullable=True),
        sa.Column("action",      sa.String(100), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("patient_id",  postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("ip_address",  sa.String(50),  nullable=True),
        sa.Column("status_code", sa.Integer(),   nullable=True),
        sa.Column("error_msg",   sa.Text(),      nullable=True),
        sa.Column("created_at",  sa.DateTime(),  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_audit_logs_hospital_id",      "audit_logs", ["hospital_id"])
    op.create_index("ix_audit_hospital_created",      "audit_logs", ["hospital_id", "created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("unit_corrections")
    op.drop_table("clinical_flags")
    op.drop_table("biomarker_readings")
    op.drop_table("analyses")
    op.drop_table("patients")
    op.drop_table("hospitals")
