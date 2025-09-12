"""
Database module for the Telematics Insurance System.

This module provides database models, repositories, and engine configuration
for the telematics insurance system.
"""

from .engine import init_db, get_session, get_session_context, reset_db
from .models import (
    Driver, Vehicle, Policy, Trip, TelematicsDataPoint, DrivingEvent,
    RiskScore, PremiumHistory, FeatureRow, LocationRiskProfile, VehicleRiskProfile,
    DriverResponse, VehicleResponse, PolicyResponse, TripResponse, RiskScoreResponse
)
from .repositories import (
    BaseRepository, DriverRepository, VehicleRepository, PolicyRepository,
    TripRepository, TelematicsDataPointRepository, DrivingEventRepository,
    RiskScoreRepository, FeatureRowRepository, LocationRiskProfileRepository,
    VehicleRiskProfileRepository
)

__all__ = [
    # Engine functions
    "init_db", "get_session", "get_session_context", "reset_db",
    
    # Models
    "Driver", "Vehicle", "Policy", "Trip", "TelematicsDataPoint", "DrivingEvent",
    "RiskScore", "PremiumHistory", "FeatureRow", "LocationRiskProfile", "VehicleRiskProfile",
    "DriverResponse", "VehicleResponse", "PolicyResponse", "TripResponse", "RiskScoreResponse",
    
    # Repositories
    "BaseRepository", "DriverRepository", "VehicleRepository", "PolicyRepository",
    "TripRepository", "TelematicsDataPointRepository", "DrivingEventRepository",
    "RiskScoreRepository", "FeatureRowRepository", "LocationRiskProfileRepository",
    "VehicleRiskProfileRepository"
]
