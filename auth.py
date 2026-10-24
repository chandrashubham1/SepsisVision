"""
auth.py — API key authentication for SepsisAI hospital API
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared

Key format:  sk-sep-<43 URL-safe base64 chars>  (≈264 bits entropy)
Storage:     SHA-256 hex digest only — raw key is shown once and never stored.
Transport:   X-API-Key request header (HTTPS mandatory in production).
"""
from __future__ import annotations
import hashlib
import logging
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

import config
from database import get_db

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


# ── Key generation & hashing ──────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key pair.
    Returns (raw_key, sha256_hex_hash).
    Show raw_key to the user ONCE at registration; store only the hash.
    """
    raw  = "sk-sep-" + secrets.token_urlsafe(32)
    hsh  = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hsh


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def api_key_prefix(raw: str) -> str:
    """
    Return the first 12 chars for display / identification without exposing the key.
    e.g.  "sk-sep-abc1"
    """
    return raw[:12]


# ── FastAPI dependencies ──────────────────────────────────────────────────────

def get_current_hospital(
    api_key: Annotated[str, Security(_api_key_header)],
    db:      Annotated[Session, Depends(get_db)],
):
    """
    FastAPI dependency — validates X-API-Key header.
    Returns the Hospital ORM object on success; raises HTTP 401 on failure.
    Inject as:  hospital = Depends(get_current_hospital)

    DEV bypass: if DEV_API_KEY is set in config and the submitted key matches,
    a mock Hospital object is returned without touching the database.
    """
    # ── Dev bypass (no DB required) ───────────────────────────────────────────
    if config.DEV_API_KEY and api_key == config.DEV_API_KEY:
        from types import SimpleNamespace
        return SimpleNamespace(
            id=0,
            name="Dev / Local",
            tier="research",
            active=True,
            api_key_hash="dev",
        )

    from db_models import Hospital
    hsh = hash_api_key(api_key)
    try:
        hospital = (
            db.query(Hospital)
            .filter(Hospital.api_key_hash == hsh, Hospital.active == True)
            .first()
        )
    except Exception:
        hospital = None
    if not hospital:
        logger.warning("API request with invalid or inactive key (prefix=%s)", api_key[:12])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key. "
                   "Contact Adnexus Biotechnologies to register your hospital.",
        )
    return hospital


def require_admin(
    api_key: Annotated[str, Security(_api_key_header)],
) -> None:
    """
    FastAPI dependency — validates the admin API key (set via ADMIN_API_KEY env var).
    Use on hospital registration / management endpoints.
    """
    if not config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key not configured on this server. "
                   "Set the ADMIN_API_KEY environment variable.",
        )
    if api_key != config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
