"""
Database models for the Telematics Insurance System.

This module defines all database tables using SQLModel for the telematics
insurance system, including drivers, vehicles, policies, trips, and analytics.
"""

from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4
import json


class DriverBase(SQLModel):
    """Base driver model with common fields."""
    name: str
    email: str
    phone: str
    license_number: str
    date_of_birth: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Driver(DriverBase, table=True):
    """Driver entity representing a policyholder."""
    __tablename__ = "drivers"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    
    # Relationships
    vehicles: List["Vehicle"] = Relationship(back_populates="driver")
    policies: List["Policy"] = Relationship(back_populates="driver")
    trips: List["Trip"] = Relationship(back_populates="driver")
    risk_scores: List["RiskScore"] = Relationship(back_populates="driver")


class VehicleBase(SQLModel):
    """Base vehicle model with common fields."""
    make: str
    model: str
    year: int
    vin: str
    license_plate: str
    engine_type: str = "gas"  # gas, hybrid, electric
    safety_rating: float = 4.0
    theft_probability: float = 0.03
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Vehicle(VehicleBase, table=True):
    """Vehicle entity representing an insured vehicle."""
    __tablename__ = "vehicles"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    driver_id: UUID = Field(foreign_key="drivers.id")
    
    # Relationships
    driver: Driver = Relationship(back_populates="vehicles")
    policies: List["Policy"] = Relationship(back_populates="vehicle")
    trips: List["Trip"] = Relationship(back_populates="vehicle")


class PolicyBase(SQLModel):
    """Base policy model with common fields."""
    policy_number: str
    start_date: datetime
    end_date: Optional[datetime] = None
    base_premium: float
    current_premium: float
    status: str = "active"  # active, suspended, cancelled
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Policy(PolicyBase, table=True):
    """Insurance policy entity."""
    __tablename__ = "policies"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    driver_id: UUID = Field(foreign_key="drivers.id")
    vehicle_id: UUID = Field(foreign_key="vehicles.id")
    
    # Relationships
    driver: Driver = Relationship(back_populates="policies")
    vehicle: Vehicle = Relationship(back_populates="policies")
    premium_history: List["PremiumHistory"] = Relationship(back_populates="policy")


class TripBase(SQLModel):
    """Base trip model with common fields."""
    start_time: datetime
    end_time: Optional[datetime] = None
    start_latitude: float
    start_longitude: float
    end_latitude: Optional[float] = None
    end_longitude: Optional[float] = None
    total_distance_km: float = 0.0
    total_duration_minutes: float = 0.0
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    overall_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Trip(TripBase, table=True):
    """Trip entity representing a driving session."""
    __tablename__ = "trips"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    driver_id: UUID = Field(foreign_key="drivers.id")
    vehicle_id: UUID = Field(foreign_key="vehicles.id")
    
    # Relationships
    driver: Driver = Relationship(back_populates="trips")
    vehicle: Vehicle = Relationship(back_populates="trips")
    telematics_data: List["TelematicsDataPoint"] = Relationship(back_populates="trip")
    driving_events: List["DrivingEvent"] = Relationship(back_populates="trip")


class TelematicsDataPointBase(SQLModel):
    """Base telematics data point model."""
    timestamp: datetime
    latitude: float
    longitude: float
    speed_kmh: float
    acceleration_ms2: float
    heading_degrees: float
    gps_accuracy_meters: float
    road_type: str = "city_street"  # highway, city_street, residential, rural, parking_lot
    weather_condition: str = "clear"  # clear, rain, snow, fog, ice
    gyroscope_x: float = 0.0
    gyroscope_y: float = 0.0
    gyroscope_z: float = 0.0
    magnetometer_x: float = 0.0
    magnetometer_y: float = 0.0
    magnetometer_z: float = 0.0
    engine_rpm: float = 0.0
    fuel_level: float = 1.0
    odometer_reading_km: float = 0.0


class TelematicsDataPoint(TelematicsDataPointBase, table=True):
    """Individual telematics data point."""
    __tablename__ = "telematics_data_points"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    trip_id: UUID = Field(foreign_key="trips.id")
    
    # Relationships
    trip: Trip = Relationship(back_populates="telematics_data")


class DrivingEventBase(SQLModel):
    """Base driving event model."""
    event_type: str  # hard_brake, hard_acceleration, speeding, sharp_turn
    timestamp: datetime
    severity: float  # 0-1 scale
    latitude: float
    longitude: float
    details: str  # JSON string with event-specific data


class DrivingEvent(DrivingEventBase, table=True):
    """Driving event entity."""
    __tablename__ = "driving_events"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    trip_id: UUID = Field(foreign_key="trips.id")
    
    # Relationships
    trip: Trip = Relationship(back_populates="driving_events")


class RiskScoreBase(SQLModel):
    """Base risk score model."""
    score: float  # 0-100 scale
    confidence: float  # 0-1 scale
    factors: str  # JSON string with risk factors
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class RiskScore(RiskScoreBase, table=True):
    """Risk score entity for a driver."""
    __tablename__ = "risk_scores"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    driver_id: UUID = Field(foreign_key="drivers.id")
    
    # Relationships
    driver: Driver = Relationship(back_populates="risk_scores")


class PremiumHistoryBase(SQLModel):
    """Base premium history model."""
    premium_amount: float
    risk_score: float
    factors_applied: str  # JSON string with pricing factors
    effective_date: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PremiumHistory(PremiumHistoryBase, table=True):
    """Premium history tracking."""
    __tablename__ = "premium_history"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    policy_id: UUID = Field(foreign_key="policies.id")
    
    # Relationships
    policy: Policy = Relationship(back_populates="premium_history")


class FeatureRowBase(SQLModel):
    """Base feature row model for ML training data."""
    driver_id: str
    distance_km: float
    speeding_pct: float
    harsh_brake_per_100km: float
    harsh_accel_per_100km: float
    night_fraction: float
    rush_hour_fraction: float
    highway_fraction: float
    city_fraction: float
    residential_fraction: float
    weather_impact_score: float
    road_quality_score: float
    average_engine_rpm: float
    fuel_efficiency_score: float
    sample_count: int
    source_version: str = "v1"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureRow(FeatureRowBase, table=True):
    """Feature row for ML model training."""
    __tablename__ = "feature_rows"
    
    trip_id: str = Field(primary_key=True)


class LocationRiskProfileBase(SQLModel):
    """Base location risk profile model."""
    location_name: str
    latitude: float
    longitude: float
    crime_rate: float  # 0-1 scale
    accident_frequency: float  # accidents per 1000 vehicles
    road_quality_score: float  # 0-1 scale
    traffic_density: float  # 0-1 scale
    speed_limit_kmh: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LocationRiskProfile(LocationRiskProfileBase, table=True):
    """Location risk profile for geographic risk assessment."""
    __tablename__ = "location_risk_profiles"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)


class VehicleRiskProfileBase(SQLModel):
    """Base vehicle risk profile model."""
    make: str
    model: str
    year: int
    engine_type: str
    safety_rating: float  # 0-5 stars
    theft_probability: float  # 0-1 scale
    age_risk_factor: float  # 0-1 scale
    performance_rating: float  # 0-1 scale
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class VehicleRiskProfile(VehicleRiskProfileBase, table=True):
    """Vehicle risk profile for vehicle-specific risk assessment."""
    __tablename__ = "vehicle_risk_profiles"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)


# Pydantic models for API responses (without table=True)
class DriverResponse(DriverBase):
    """Driver response model for API."""
    id: UUID


class VehicleResponse(VehicleBase):
    """Vehicle response model for API."""
    id: UUID
    driver_id: UUID


class PolicyResponse(PolicyBase):
    """Policy response model for API."""
    id: UUID
    driver_id: UUID
    vehicle_id: UUID


class TripResponse(TripBase):
    """Trip response model for API."""
    id: UUID
    driver_id: UUID
    vehicle_id: UUID


class RiskScoreResponse(RiskScoreBase):
    """Risk score response model for API."""
    id: UUID
    driver_id: UUID
