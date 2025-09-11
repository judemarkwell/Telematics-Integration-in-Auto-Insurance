"""
Domain entities for the telematics insurance system.

These entities represent the core business objects with identity and lifecycle.
Following DDD principles, they encapsulate business rules and state transitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4


@dataclass
class Driver:
    """Driver entity representing a policyholder."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    email: str = ""
    phone: str = ""
    license_number: str = ""
    date_of_birth: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_profile(self, name: str = None, email: str = None, phone: str = None) -> None:
        """Update driver profile information."""
        if name is not None:
            self.name = name
        if email is not None:
            self.email = email
        if phone is not None:
            self.phone = phone
        self.updated_at = datetime.utcnow()


@dataclass
class Vehicle:
    """Vehicle entity representing an insured vehicle."""
    id: UUID = field(default_factory=uuid4)
    driver_id: UUID = field(default_factory=uuid4)
    make: str = ""
    model: str = ""
    year: int = 0
    vin: str = ""
    license_plate: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_valid_year(self) -> bool:
        """Check if vehicle year is valid."""
        current_year = datetime.now().year
        return 1900 <= self.year <= current_year + 1


@dataclass
class Policy:
    """Insurance policy entity."""
    id: UUID = field(default_factory=uuid4)
    driver_id: UUID = field(default_factory=uuid4)
    vehicle_id: UUID = field(default_factory=uuid4)
    policy_number: str = ""
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: Optional[datetime] = None
    base_premium: float = 0.0
    current_premium: float = 0.0
    status: str = "active"  # active, suspended, cancelled
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_premium(self, new_premium: float) -> None:
        """Update policy premium."""
        if new_premium < 0:
            raise ValueError("Premium cannot be negative")
        self.current_premium = new_premium
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if policy is currently active."""
        now = datetime.utcnow()
        return (self.status == "active" and 
                self.start_date <= now and 
                (self.end_date is None or self.end_date > now))


@dataclass
class RiskScore:
    """Risk score entity for a driver."""
    id: UUID = field(default_factory=uuid4)
    driver_id: UUID = field(default_factory=uuid4)
    score: float = 0.0  # 0-100 scale
    confidence: float = 0.0  # 0-1 scale
    factors: List[str] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_high_risk(self, threshold: float = 70.0) -> bool:
        """Check if driver is considered high risk."""
        return self.score >= threshold
    
    def get_risk_category(self) -> str:
        """Get risk category based on score."""
        if self.score < 30:
            return "low"
        elif self.score < 70:
            return "medium"
        elif self.score < 85:
            return "high"
        else:
            return "very_high"
