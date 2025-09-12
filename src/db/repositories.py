"""
Repository classes for database operations.

This module provides repository classes that encapsulate database operations
for each entity, following the Repository pattern for clean separation of
data access logic.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlmodel import Session, select, and_, or_, func
from .models import (
    Driver, Vehicle, Policy, Trip, TelematicsDataPoint, DrivingEvent,
    RiskScore, PremiumHistory, FeatureRow, LocationRiskProfile, VehicleRiskProfile
)
from .engine import get_session_context


class BaseRepository:
    """Base repository class with common database operations."""
    
    def __init__(self, model_class):
        self.model_class = model_class
    
    def create(self, session: Session, **kwargs) -> Any:
        """Create a new entity."""
        entity = self.model_class(**kwargs)
        session.add(entity)
        session.commit()
        session.refresh(entity)
        return entity
    
    def get_by_id(self, session: Session, entity_id: UUID) -> Optional[Any]:
        """Get entity by ID."""
        return session.get(self.model_class, entity_id)
    
    def get_all(self, session: Session, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all entities with pagination."""
        statement = select(self.model_class).offset(offset).limit(limit)
        return session.exec(statement).all()
    
    def update(self, session: Session, entity: Any, **kwargs) -> Any:
        """Update an entity."""
        for key, value in kwargs.items():
            if hasattr(entity, key):
                setattr(entity, key, value)
        session.add(entity)
        session.commit()
        session.refresh(entity)
        return entity
    
    def delete(self, session: Session, entity_id: UUID) -> bool:
        """Delete an entity by ID."""
        entity = self.get_by_id(session, entity_id)
        if entity:
            session.delete(entity)
            session.commit()
            return True
        return False


class DriverRepository(BaseRepository):
    """Repository for Driver entities."""
    
    def __init__(self):
        super().__init__(Driver)
    
    def get_by_email(self, session: Session, email: str) -> Optional[Driver]:
        """Get driver by email address."""
        statement = select(Driver).where(Driver.email == email)
        return session.exec(statement).first()
    
    def get_by_license(self, session: Session, license_number: str) -> Optional[Driver]:
        """Get driver by license number."""
        statement = select(Driver).where(Driver.license_number == license_number)
        return session.exec(statement).first()
    
    def search_by_name(self, session: Session, name: str) -> List[Driver]:
        """Search drivers by name (partial match)."""
        statement = select(Driver).where(Driver.name.ilike(f"%{name}%"))
        return session.exec(statement).all()


class VehicleRepository(BaseRepository):
    """Repository for Vehicle entities."""
    
    def __init__(self):
        super().__init__(Vehicle)
    
    def get_by_driver(self, session: Session, driver_id: UUID) -> List[Vehicle]:
        """Get all vehicles for a driver."""
        statement = select(Vehicle).where(Vehicle.driver_id == driver_id)
        return session.exec(statement).all()
    
    def get_by_vin(self, session: Session, vin: str) -> Optional[Vehicle]:
        """Get vehicle by VIN."""
        statement = select(Vehicle).where(Vehicle.vin == vin)
        return session.exec(statement).first()
    
    def get_by_license_plate(self, session: Session, license_plate: str) -> Optional[Vehicle]:
        """Get vehicle by license plate."""
        statement = select(Vehicle).where(Vehicle.license_plate == license_plate)
        return session.exec(statement).first()


class PolicyRepository(BaseRepository):
    """Repository for Policy entities."""
    
    def __init__(self):
        super().__init__(Policy)
    
    def get_by_driver(self, session: Session, driver_id: UUID) -> List[Policy]:
        """Get all policies for a driver."""
        statement = select(Policy).where(Policy.driver_id == driver_id)
        return session.exec(statement).all()
    
    def get_by_vehicle(self, session: Session, vehicle_id: UUID) -> List[Policy]:
        """Get all policies for a vehicle."""
        statement = select(Policy).where(Policy.vehicle_id == vehicle_id)
        return session.exec(statement).all()
    
    def get_active_policies(self, session: Session) -> List[Policy]:
        """Get all active policies."""
        statement = select(Policy).where(Policy.status == "active")
        return session.exec(statement).all()
    
    def get_by_policy_number(self, session: Session, policy_number: str) -> Optional[Policy]:
        """Get policy by policy number."""
        statement = select(Policy).where(Policy.policy_number == policy_number)
        return session.exec(statement).first()


class TripRepository(BaseRepository):
    """Repository for Trip entities."""
    
    def __init__(self):
        super().__init__(Trip)
    
    def get_by_driver(self, session: Session, driver_id: UUID, limit: int = 50) -> List[Trip]:
        """Get trips for a driver."""
        statement = (select(Trip)
                    .where(Trip.driver_id == driver_id)
                    .order_by(Trip.start_time.desc())
                    .limit(limit))
        return session.exec(statement).all()
    
    def get_by_vehicle(self, session: Session, vehicle_id: UUID, limit: int = 50) -> List[Trip]:
        """Get trips for a vehicle."""
        statement = (select(Trip)
                    .where(Trip.vehicle_id == vehicle_id)
                    .order_by(Trip.start_time.desc())
                    .limit(limit))
        return session.exec(statement).all()
    
    def get_recent_trips(self, session: Session, days: int = 30) -> List[Trip]:
        """Get recent trips within specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        statement = (select(Trip)
                    .where(Trip.start_time >= cutoff_date)
                    .order_by(Trip.start_time.desc()))
        return session.exec(statement).all()
    
    def get_trips_in_date_range(self, session: Session, start_date: datetime, 
                               end_date: datetime) -> List[Trip]:
        """Get trips within a date range."""
        statement = (select(Trip)
                    .where(and_(Trip.start_time >= start_date, Trip.start_time <= end_date))
                    .order_by(Trip.start_time.desc()))
        return session.exec(statement).all()


class TelematicsDataPointRepository(BaseRepository):
    """Repository for TelematicsDataPoint entities."""
    
    def __init__(self):
        super().__init__(TelematicsDataPoint)
    
    def get_by_trip(self, session: Session, trip_id: UUID) -> List[TelematicsDataPoint]:
        """Get all telematics data points for a trip."""
        statement = (select(TelematicsDataPoint)
                    .where(TelematicsDataPoint.trip_id == trip_id)
                    .order_by(TelematicsDataPoint.timestamp))
        return session.exec(statement).all()
    
    def get_by_trip_time_range(self, session: Session, trip_id: UUID, 
                              start_time: datetime, end_time: datetime) -> List[TelematicsDataPoint]:
        """Get telematics data points for a trip within time range."""
        statement = (select(TelematicsDataPoint)
                    .where(and_(
                        TelematicsDataPoint.trip_id == trip_id,
                        TelematicsDataPoint.timestamp >= start_time,
                        TelematicsDataPoint.timestamp <= end_time
                    ))
                    .order_by(TelematicsDataPoint.timestamp))
        return session.exec(statement).all()
    
    def get_speeding_events(self, session: Session, trip_id: UUID, 
                           speed_threshold: float = 80.0) -> List[TelematicsDataPoint]:
        """Get data points where speed exceeded threshold."""
        statement = (select(TelematicsDataPoint)
                    .where(and_(
                        TelematicsDataPoint.trip_id == trip_id,
                        TelematicsDataPoint.speed_kmh > speed_threshold
                    ))
                    .order_by(TelematicsDataPoint.timestamp))
        return session.exec(statement).all()


class DrivingEventRepository(BaseRepository):
    """Repository for DrivingEvent entities."""
    
    def __init__(self):
        super().__init__(DrivingEvent)
    
    def get_by_trip(self, session: Session, trip_id: UUID) -> List[DrivingEvent]:
        """Get all driving events for a trip."""
        statement = (select(DrivingEvent)
                    .where(DrivingEvent.trip_id == trip_id)
                    .order_by(DrivingEvent.timestamp))
        return session.exec(statement).all()
    
    def get_by_event_type(self, session: Session, event_type: str, 
                         limit: int = 100) -> List[DrivingEvent]:
        """Get events by type."""
        statement = (select(DrivingEvent)
                    .where(DrivingEvent.event_type == event_type)
                    .order_by(DrivingEvent.timestamp.desc())
                    .limit(limit))
        return session.exec(statement).all()
    
    def get_events_by_severity(self, session: Session, min_severity: float = 0.5) -> List[DrivingEvent]:
        """Get events above severity threshold."""
        statement = (select(DrivingEvent)
                    .where(DrivingEvent.severity >= min_severity)
                    .order_by(DrivingEvent.severity.desc()))
        return session.exec(statement).all()


class RiskScoreRepository(BaseRepository):
    """Repository for RiskScore entities."""
    
    def __init__(self):
        super().__init__(RiskScore)
    
    def get_by_driver(self, session: Session, driver_id: UUID) -> List[RiskScore]:
        """Get risk scores for a driver."""
        statement = (select(RiskScore)
                    .where(RiskScore.driver_id == driver_id)
                    .order_by(RiskScore.calculated_at.desc()))
        return session.exec(statement).all()
    
    def get_latest_by_driver(self, session: Session, driver_id: UUID) -> Optional[RiskScore]:
        """Get the latest risk score for a driver."""
        statement = (select(RiskScore)
                    .where(RiskScore.driver_id == driver_id)
                    .order_by(RiskScore.calculated_at.desc())
                    .limit(1))
        return session.exec(statement).first()
    
    def get_high_risk_drivers(self, session: Session, threshold: float = 70.0) -> List[RiskScore]:
        """Get drivers with high risk scores."""
        statement = (select(RiskScore)
                    .where(RiskScore.score >= threshold)
                    .order_by(RiskScore.score.desc()))
        return session.exec(statement).all()


class FeatureRowRepository(BaseRepository):
    """Repository for FeatureRow entities."""
    
    def __init__(self):
        super().__init__(FeatureRow)
    
    def get_by_driver(self, session: Session, driver_id: str, limit: int = 100) -> List[FeatureRow]:
        """Get feature rows for a driver."""
        statement = (select(FeatureRow)
                    .where(FeatureRow.driver_id == driver_id)
                    .order_by(FeatureRow.created_at.desc())
                    .limit(limit))
        return session.exec(statement).all()
    
    def get_training_data(self, session: Session, limit: int = 1000) -> List[FeatureRow]:
        """Get feature rows for ML training."""
        statement = (select(FeatureRow)
                    .order_by(FeatureRow.created_at.desc())
                    .limit(limit))
        return session.exec(statement).all()
    
    def get_recent_features(self, session: Session, days: int = 30) -> List[FeatureRow]:
        """Get recent feature rows."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        statement = (select(FeatureRow)
                    .where(FeatureRow.created_at >= cutoff_date)
                    .order_by(FeatureRow.created_at.desc()))
        return session.exec(statement).all()


class LocationRiskProfileRepository(BaseRepository):
    """Repository for LocationRiskProfile entities."""
    
    def __init__(self):
        super().__init__(LocationRiskProfile)
    
    def get_by_location(self, session: Session, latitude: float, longitude: float, 
                       tolerance: float = 0.01) -> Optional[LocationRiskProfile]:
        """Get location profile by coordinates (with tolerance)."""
        statement = (select(LocationRiskProfile)
                    .where(and_(
                        abs(LocationRiskProfile.latitude - latitude) <= tolerance,
                        abs(LocationRiskProfile.longitude - longitude) <= tolerance
                    )))
        return session.exec(statement).first()
    
    def get_high_risk_locations(self, session: Session, 
                               crime_threshold: float = 0.5) -> List[LocationRiskProfile]:
        """Get high-risk locations."""
        statement = (select(LocationRiskProfile)
                    .where(LocationRiskProfile.crime_rate >= crime_threshold)
                    .order_by(LocationRiskProfile.crime_rate.desc()))
        return session.exec(statement).all()


class VehicleRiskProfileRepository(BaseRepository):
    """Repository for VehicleRiskProfile entities."""
    
    def __init__(self):
        super().__init__(VehicleRiskProfile)
    
    def get_by_make_model(self, session: Session, make: str, model: str) -> Optional[VehicleRiskProfile]:
        """Get vehicle profile by make and model."""
        statement = (select(VehicleRiskProfile)
                    .where(and_(
                        VehicleRiskProfile.make == make,
                        VehicleRiskProfile.model == model
                    )))
        return session.exec(statement).first()
    
    def get_high_theft_risk_vehicles(self, session: Session, 
                                   theft_threshold: float = 0.05) -> List[VehicleRiskProfile]:
        """Get vehicles with high theft risk."""
        statement = (select(VehicleRiskProfile)
                    .where(VehicleRiskProfile.theft_probability >= theft_threshold)
                    .order_by(VehicleRiskProfile.theft_probability.desc()))
        return session.exec(statement).all()
