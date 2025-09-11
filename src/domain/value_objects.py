"""
Value objects for the telematics insurance system.

These are immutable objects that represent meaningful values without identity.
Following DDD principles, they encapsulate domain concepts and validation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import math


@dataclass(frozen=True)
class Coordinate:
    """GPS coordinate value object."""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        """Validate coordinate values."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")
    
    def distance_to(self, other: "Coordinate") -> float:
        """Calculate distance to another coordinate in kilometers using Haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(self.latitude)
        lat2_rad = math.radians(other.latitude)
        delta_lat = math.radians(other.latitude - self.latitude)
        delta_lon = math.radians(other.longitude - self.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


@dataclass(frozen=True)
class Speed:
    """Speed value object in km/h."""
    value: float
    
    def __post_init__(self):
        """Validate speed value."""
        if self.value < 0:
            raise ValueError(f"Speed cannot be negative, got {self.value}")
        if self.value > 300:  # Reasonable maximum speed
            raise ValueError(f"Speed seems unrealistic, got {self.value} km/h")
    
    def to_mph(self) -> float:
        """Convert to miles per hour."""
        return self.value * 0.621371
    
    def is_speeding(self, speed_limit: float) -> bool:
        """Check if speed exceeds the limit."""
        return self.value > speed_limit


@dataclass(frozen=True)
class Acceleration:
    """Acceleration value object in m/s²."""
    value: float
    
    def __post_init__(self):
        """Validate acceleration value."""
        if abs(self.value) > 20:  # Reasonable maximum acceleration
            raise ValueError(f"Acceleration seems unrealistic, got {self.value} m/s²")
    
    def is_hard_braking(self, threshold: float = -3.0) -> bool:
        """Check if this represents hard braking."""
        return self.value <= threshold
    
    def is_hard_acceleration(self, threshold: float = 3.0) -> bool:
        """Check if this represents hard acceleration."""
        return self.value >= threshold


@dataclass(frozen=True)
class TimeOfDay:
    """Time of day value object."""
    hour: int
    minute: int = 0
    second: int = 0
    
    def __post_init__(self):
        """Validate time values."""
        if not (0 <= self.hour <= 23):
            raise ValueError(f"Hour must be between 0 and 23, got {self.hour}")
        if not (0 <= self.minute <= 59):
            raise ValueError(f"Minute must be between 0 and 59, got {self.minute}")
        if not (0 <= self.second <= 59):
            raise ValueError(f"Second must be between 0 and 59, got {self.second}")
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> "TimeOfDay":
        """Create from datetime object."""
        return cls(hour=dt.hour, minute=dt.minute, second=dt.second)
    
    def is_rush_hour(self) -> bool:
        """Check if this is during rush hour (7-9 AM or 5-7 PM)."""
        return (7 <= self.hour <= 9) or (17 <= self.hour <= 19)
    
    def is_night_time(self) -> bool:
        """Check if this is during night time (10 PM - 6 AM)."""
        return self.hour >= 22 or self.hour <= 6


@dataclass(frozen=True)
class Distance:
    """Distance value object in kilometers."""
    value: float
    
    def __post_init__(self):
        """Validate distance value."""
        if self.value < 0:
            raise ValueError(f"Distance cannot be negative, got {self.value}")
    
    def to_miles(self) -> float:
        """Convert to miles."""
        return self.value * 0.621371
    
    def to_meters(self) -> float:
        """Convert to meters."""
        return self.value * 1000


@dataclass(frozen=True)
class Premium:
    """Premium value object in currency."""
    amount: float
    currency: str = "USD"
    
    def __post_init__(self):
        """Validate premium amount."""
        if self.amount < 0:
            raise ValueError(f"Premium amount cannot be negative, got {self.amount}")
    
    def apply_discount(self, percentage: float) -> "Premium":
        """Apply discount percentage and return new premium."""
        if not (0 <= percentage <= 100):
            raise ValueError(f"Discount percentage must be between 0 and 100, got {percentage}")
        discounted_amount = self.amount * (1 - percentage / 100)
        return Premium(amount=discounted_amount, currency=self.currency)
    
    def apply_surcharge(self, percentage: float) -> "Premium":
        """Apply surcharge percentage and return new premium."""
        if percentage < 0:
            raise ValueError(f"Surcharge percentage cannot be negative, got {percentage}")
        surcharged_amount = self.amount * (1 + percentage / 100)
        return Premium(amount=surcharged_amount, currency=self.currency)

