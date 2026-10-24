"""Add Module B columns to analyses

Adds three columns supporting the KILL-4 reduced-panel BEI engine (Module B):
  - module      : which engine was run ('A', 'B', or 'BOTH'); defaults to 'A'
  - bei_b_score : Module B BEI score (nullable — null when module='A')
  - bei_b_zone  : Module B BEI zone  (nullable — null when module='A')

Also adds a composite index on (hospital_id, bei_b_zone) to support
zone-based cohort queries on Module B results.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23 00:00:00.000000
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op

# ── Revision identifiers ──────────────────────────────────────────────────────
revision      = "0002"
down_revision = "0001"
branch_labels = None
depends_on    = None


def upgrade() -> None:
    # Add Module B columns ─────────────────────────────────────────────────────
    op.add_column(
        "analyses",
        sa.Column("module", sa.String(10), nullable=False, server_default="A"),
    )
    op.add_column(
        "analyses",
        sa.Column("bei_b_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "analyses",
        sa.Column("bei_b_zone", sa.String(20), nullable=True),
    )

    # Composite index for Module B zone queries ─────────────────────────────────
    op.create_index(
        "ix_analyses_hospital_b_zone",
        "analyses",
        ["hospital_id", "bei_b_zone"],
    )


def downgrade() -> None:
    op.drop_index("ix_analyses_hospital_b_zone", table_name="analyses")
    op.drop_column("analyses", "bei_b_zone")
    op.drop_column("analyses", "bei_b_score")
    op.drop_column("analyses", "module")
