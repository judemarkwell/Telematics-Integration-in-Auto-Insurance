"""
Factory for creating realistic vehicle and location profiles for telematics simulation.

This module provides predefined profiles and utilities for generating realistic
vehicle characteristics and location risk factors for enhanced simulation.
"""

import random
from typing import Dict, List
from .telematics_simulator import VehicleProfile, LocationRiskProfile


class VehicleProfileFactory:
    """Factory for creating realistic vehicle profiles."""
    
    # Realistic vehicle data based on common makes/models
    VEHICLE_DATABASE = {
        "Toyota": {
            "Camry": {"safety_rating": 4.5, "theft_probability": 0.02, "performance": 0.6},
            "Corolla": {"safety_rating": 4.3, "theft_probability": 0.03, "performance": 0.5},
            "Prius": {"safety_rating": 4.4, "theft_probability": 0.01, "performance": 0.4},
            "RAV4": {"safety_rating": 4.6, "theft_probability": 0.025, "performance": 0.7},
            "Highlander": {"safety_rating": 4.7, "theft_probability": 0.02, "performance": 0.8}
        },
        "Honda": {
            "Civic": {"safety_rating": 4.4, "theft_probability": 0.04, "performance": 0.6},
            "Accord": {"safety_rating": 4.5, "theft_probability": 0.03, "performance": 0.7},
            "CR-V": {"safety_rating": 4.6, "theft_probability": 0.025, "performance": 0.7},
            "Pilot": {"safety_rating": 4.7, "theft_probability": 0.02, "performance": 0.8}
        },
        "Ford": {
            "F-150": {"safety_rating": 4.3, "theft_probability": 0.05, "performance": 0.8},
            "Escape": {"safety_rating": 4.4, "theft_probability": 0.03, "performance": 0.6},
            "Explorer": {"safety_rating": 4.5, "theft_probability": 0.03, "performance": 0.7},
            "Mustang": {"safety_rating": 4.2, "theft_probability": 0.08, "performance": 0.9}
        },
        "BMW": {
            "3 Series": {"safety_rating": 4.6, "theft_probability": 0.06, "performance": 0.8},
            "5 Series": {"safety_rating": 4.7, "theft_probability": 0.05, "performance": 0.9},
            "X3": {"safety_rating": 4.5, "theft_probability": 0.04, "performance": 0.8},
            "X5": {"safety_rating": 4.6, "theft_probability": 0.04, "performance": 0.9}
        },
        "Tesla": {
            "Model 3": {"safety_rating": 4.8, "theft_probability": 0.01, "performance": 0.9},
            "Model S": {"safety_rating": 4.9, "theft_probability": 0.01, "performance": 0.95},
            "Model Y": {"safety_rating": 4.7, "theft_probability": 0.01, "performance": 0.9}
        }
    }
    
    @classmethod
    def create_random_vehicle(cls, year_range: tuple = (2015, 2024)) -> VehicleProfile:
        """Create a random vehicle profile with realistic characteristics."""
        make = random.choice(list(cls.VEHICLE_DATABASE.keys()))
        model = random.choice(list(cls.VEHICLE_DATABASE[make].keys()))
        year = random.randint(year_range[0], year_range[1])
        
        vehicle_data = cls.VEHICLE_DATABASE[make][model]
        
        # Calculate age risk factor (older cars = higher risk)
        current_year = 2024
        age = current_year - year
        age_risk_factor = min(0.8, age * 0.05)  # Max 0.8 risk for very old cars
        
        # Determine engine type based on make/model
        engine_type = "electric" if make == "Tesla" else random.choice(["gas", "hybrid"])
        
        return VehicleProfile(
            make=make,
            model=model,
            year=year,
            engine_type=engine_type,
            safety_rating=vehicle_data["safety_rating"],
            theft_probability=vehicle_data["theft_probability"],
            age_risk_factor=age_risk_factor,
            performance_rating=vehicle_data["performance"]
        )
    
    @classmethod
    def create_vehicle_by_specs(
        cls, 
        make: str, 
        model: str, 
        year: int, 
        engine_type: str = "gas"
    ) -> VehicleProfile:
        """Create a vehicle profile with specific characteristics."""
        if make in cls.VEHICLE_DATABASE and model in cls.VEHICLE_DATABASE[make]:
            vehicle_data = cls.VEHICLE_DATABASE[make][model]
        else:
            # Default values for unknown vehicles
            vehicle_data = {
                "safety_rating": 4.0,
                "theft_probability": 0.03,
                "performance": 0.6
            }
        
        # Calculate age risk factor
        current_year = 2024
        age = current_year - year
        age_risk_factor = min(0.8, age * 0.05)
        
        return VehicleProfile(
            make=make,
            model=model,
            year=year,
            engine_type=engine_type,
            safety_rating=vehicle_data["safety_rating"],
            theft_probability=vehicle_data["theft_probability"],
            age_risk_factor=age_risk_factor,
            performance_rating=vehicle_data["performance"]
        )


class LocationProfileFactory:
    """Factory for creating realistic location risk profiles."""
    
    # Realistic location data based on US city characteristics
    LOCATION_DATABASE = {
        "urban_high_risk": {
            "crime_rate": 0.7,
            "accident_frequency": 4.5,
            "road_quality_score": 0.5,
            "traffic_density": 0.9,
            "speed_limit": 35.0
        },
        "urban_medium_risk": {
            "crime_rate": 0.4,
            "accident_frequency": 3.0,
            "road_quality_score": 0.6,
            "traffic_density": 0.7,
            "speed_limit": 40.0
        },
        "suburban": {
            "crime_rate": 0.2,
            "accident_frequency": 2.0,
            "road_quality_score": 0.8,
            "traffic_density": 0.4,
            "speed_limit": 50.0
        },
        "rural": {
            "crime_rate": 0.1,
            "accident_frequency": 1.5,
            "road_quality_score": 0.6,
            "traffic_density": 0.1,
            "speed_limit": 70.0
        },
        "highway": {
            "crime_rate": 0.05,
            "accident_frequency": 1.8,
            "road_quality_score": 0.9,
            "traffic_density": 0.6,
            "speed_limit": 100.0
        }
    }
    
    @classmethod
    def create_location_by_type(cls, location_type: str) -> LocationRiskProfile:
        """Create a location profile by predefined type."""
        if location_type not in cls.LOCATION_DATABASE:
            location_type = "suburban"  # Default fallback
        
        data = cls.LOCATION_DATABASE[location_type]
        return LocationRiskProfile(
            crime_rate=data["crime_rate"],
            accident_frequency=data["accident_frequency"],
            road_quality_score=data["road_quality_score"],
            traffic_density=data["traffic_density"],
            speed_limit=data["speed_limit"]
        )
    
    @classmethod
    def create_random_location(cls) -> LocationRiskProfile:
        """Create a random location profile."""
        location_type = random.choice(list(cls.LOCATION_DATABASE.keys()))
        return cls.create_location_by_type(location_type)
    
    @classmethod
    def create_custom_location(
        cls,
        crime_rate: float,
        accident_frequency: float,
        road_quality_score: float,
        traffic_density: float,
        speed_limit: float
    ) -> LocationRiskProfile:
        """Create a custom location profile with specific values."""
        return LocationRiskProfile(
            crime_rate=crime_rate,
            accident_frequency=accident_frequency,
            road_quality_score=road_quality_score,
            traffic_density=traffic_density,
            speed_limit=speed_limit
        )


class ProfileManager:
    """Manager for creating and storing vehicle and location profiles."""
    
    def __init__(self):
        self.vehicle_profiles: Dict[str, VehicleProfile] = {}
        self.location_profiles: Dict[str, LocationRiskProfile] = {}
    
    def create_sample_vehicles(self, count: int = 10) -> Dict[str, VehicleProfile]:
        """Create a set of sample vehicles for testing."""
        vehicles = {}
        for i in range(count):
            vehicle_id = f"vehicle_{i+1:03d}"
            vehicles[vehicle_id] = VehicleProfileFactory.create_random_vehicle()
        self.vehicle_profiles.update(vehicles)
        return vehicles
    
    def create_sample_locations(self) -> Dict[str, LocationRiskProfile]:
        """Create a set of sample locations for testing."""
        locations = {}
        for location_type in LocationProfileFactory.LOCATION_DATABASE.keys():
            location_id = f"location_{location_type}"
            locations[location_id] = LocationProfileFactory.create_location_by_type(location_type)
        self.location_profiles.update(locations)
        return locations
    
    def get_vehicle_profile(self, vehicle_id: str) -> VehicleProfile:
        """Get a vehicle profile by ID."""
        return self.vehicle_profiles.get(vehicle_id)
    
    def get_location_profile(self, location_id: str) -> LocationRiskProfile:
        """Get a location profile by ID."""
        return self.location_profiles.get(location_id)
    
    def register_vehicle(self, vehicle_id: str, profile: VehicleProfile) -> None:
        """Register a vehicle profile."""
        self.vehicle_profiles[vehicle_id] = profile
    
    def register_location(self, location_id: str, profile: LocationRiskProfile) -> None:
        """Register a location profile."""
        self.location_profiles[location_id] = profile
