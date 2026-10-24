"""
database.py — SQLAlchemy engine, session factory, and Base class
Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared
"""
from __future__ import annotations
import logging
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

import config

logger = logging.getLogger(__name__)

engine = create_engine(
    config.DATABASE_URL,
    pool_pre_ping=True,    # verify connections are alive before use
    pool_size=10,
    max_overflow=20,
    echo=config.DB_ECHO,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a database session and closes it after the request.

    Usage:
        @app.get("/foo")
        def foo(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """
    Create all tables from ORM metadata.
    Suitable for development; use Alembic migrations in production.
    """
    import db_models  # noqa: F401 — registers all models with Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created / verified.")


def check_db_connection() -> bool:
    """Probe the database connection — used in /health endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
        return False
