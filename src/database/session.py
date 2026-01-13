"""
Database Session Management
SQLAlchemy session factory and context manager
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from src.config import get_settings
from src.database.models import Base


def get_engine():
    """Create SQLAlchemy engine from settings"""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True
    )


def get_session_factory():
    """Create session factory"""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


SessionLocal = get_session_factory()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Ensures proper cleanup on exceptions.
    
    Usage:
        with get_db_session() as db:
            db.query(Patient).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Initialize database - create all tables"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (use with caution!)"""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
