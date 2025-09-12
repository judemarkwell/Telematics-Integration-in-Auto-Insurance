"""
Database engine and session management for the Telematics Insurance System.

This module provides database connection, session management, and initialization
for the telematics insurance system using SQLModel and SQLAlchemy.
"""

import os
from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

# Database configuration
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "telematics_insurance.db"
DB_PATH.parent.mkdir(exist_ok=True)  # ensure /data exists

# Support both SQLite (development) and PostgreSQL (production)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Engine configuration
engine_kwargs = {
    "echo": os.getenv("DB_ECHO", "false").lower() == "true",  # SQL logging
    "pool_pre_ping": True,  # Verify connections before use
}

# Add PostgreSQL-specific settings if using PostgreSQL
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs.update({
        "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
        "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
        "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600")),
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)


def init_db() -> None:
    """Create all tables (run once on startup)."""
    print(f"🗄️  Initializing database at: {DATABASE_URL}")
    SQLModel.metadata.create_all(engine)
    print("✅ Database tables created successfully")


def get_session() -> Session:
    """Get a new database session."""
    return Session(engine)


@contextmanager
def get_session_context() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_db() -> None:
    """Reset the database by dropping and recreating all tables."""
    print("⚠️  Resetting database - all data will be lost!")
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    print("✅ Database reset complete")


# Import all models to ensure they're registered with SQLModel
from .models import (
    Driver, Vehicle, Policy, Trip, TelematicsDataPoint, DrivingEvent,
    RiskScore, PremiumHistory, FeatureRow, LocationRiskProfile, VehicleRiskProfile
)