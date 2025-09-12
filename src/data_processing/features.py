"""
Feature extraction and ML training data preparation.

This module handles the conversion of telematics data into features
suitable for machine learning model training.
"""

from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from ..db.models import FeatureRow, Trip, TelematicsDataPoint, DrivingEvent


class FeatureExtractor:
    """Extracts features from telematics data for ML training."""
    
    def __init__(self):
        self.feature_cache: Dict[str, Dict[str, Any]] = {}
    
    def extract_trip_features(self, trip: Trip, telematics_data: List[TelematicsDataPoint], 
                            driving_events: List[DrivingEvent]) -> FeatureRow:
        """Extract comprehensive features from a trip for ML training."""
        
        # Basic trip metrics
        distance_km = trip.total_distance_km
        duration_hours = trip.total_duration_minutes / 60.0 if trip.total_duration_minutes > 0 else 0.001
        
        # Speed-based features
        speeding_events = [e for e in driving_events if e.event_type == "speeding"]
        speeding_pct = (len(speeding_events) / len(telematics_data) * 100) if telematics_data else 0
        
        # Event-based features (per 100km)
        harsh_brake_events = [e for e in driving_events if e.event_type == "hard_brake"]
        harsh_accel_events = [e for e in driving_events if e.event_type == "hard_acceleration"]
        
        harsh_brake_per_100km = (len(harsh_brake_events) / distance_km * 100) if distance_km > 0 else 0
        harsh_accel_per_100km = (len(harsh_accel_events) / distance_km * 100) if distance_km > 0 else 0
        
        # Time-based features
        night_fraction = self._calculate_night_fraction(telematics_data)
        rush_hour_fraction = self._calculate_rush_hour_fraction(telematics_data)
        
        # Road type features
        road_type_fractions = self._calculate_road_type_fractions(telematics_data)
        
        # Weather and road quality features
        weather_impact_score = self._calculate_weather_impact_score(telematics_data)
        road_quality_score = self._calculate_road_quality_score(telematics_data)
        
        # Engine and efficiency features
        average_engine_rpm = self._calculate_average_engine_rpm(telematics_data)
        fuel_efficiency_score = self._calculate_fuel_efficiency_score(telematics_data, distance_km)
        
        return FeatureRow(
            trip_id=str(trip.id),
            driver_id=str(trip.driver_id),
            distance_km=distance_km,
            speeding_pct=speeding_pct,
            harsh_brake_per_100km=harsh_brake_per_100km,
            harsh_accel_per_100km=harsh_accel_per_100km,
            night_fraction=night_fraction,
            rush_hour_fraction=rush_hour_fraction,
            highway_fraction=road_type_fractions.get("highway", 0.0),
            city_fraction=road_type_fractions.get("city_street", 0.0),
            residential_fraction=road_type_fractions.get("residential", 0.0),
            weather_impact_score=weather_impact_score,
            road_quality_score=road_quality_score,
            average_engine_rpm=average_engine_rpm,
            fuel_efficiency_score=fuel_efficiency_score,
            sample_count=len(telematics_data),
            source_version="v2"
        )
    
    def _calculate_night_fraction(self, telematics_data: List[TelematicsDataPoint]) -> float:
        """Calculate fraction of trip during night time (10 PM - 6 AM)."""
        if not telematics_data:
            return 0.0
        
        night_points = 0
        for point in telematics_data:
            hour = point.timestamp.hour
            if hour >= 22 or hour <= 6:
                night_points += 1
        
        return night_points / len(telematics_data)
    
    def _calculate_rush_hour_fraction(self, telematics_data: List[TelematicsDataPoint]) -> float:
        """Calculate fraction of trip during rush hour (7-9 AM, 5-7 PM)."""
        if not telematics_data:
            return 0.0
        
        rush_hour_points = 0
        for point in telematics_data:
            hour = point.timestamp.hour
            if (7 <= hour <= 9) or (17 <= hour <= 19):
                rush_hour_points += 1
        
        return rush_hour_points / len(telematics_data)
    
    def _calculate_road_type_fractions(self, telematics_data: List[TelematicsDataPoint]) -> Dict[str, float]:
        """Calculate fractions of trip on different road types."""
        if not telematics_data:
            return {"highway": 0.0, "city_street": 0.0, "residential": 0.0}
        
        road_type_counts = {}
        for point in telematics_data:
            road_type = point.road_type
            road_type_counts[road_type] = road_type_counts.get(road_type, 0) + 1
        
        total_points = len(telematics_data)
        return {
            road_type: count / total_points 
            for road_type, count in road_type_counts.items()
        }
    
    def _calculate_weather_impact_score(self, telematics_data: List[TelematicsDataPoint]) -> float:
        """Calculate weather impact score (0-100, higher is better)."""
        if not telematics_data:
            return 100.0
        
        weather_impacts = {
            "clear": 100.0,
            "rain": 80.0,
            "snow": 60.0,
            "fog": 70.0,
            "ice": 40.0
        }
        
        total_score = sum(weather_impacts.get(point.weather_condition, 100.0) for point in telematics_data)
        return total_score / len(telematics_data)
    
    def _calculate_road_quality_score(self, telematics_data: List[TelematicsDataPoint]) -> float:
        """Calculate road quality score based on driving smoothness."""
        if len(telematics_data) < 2:
            return 100.0
        
        # Calculate acceleration variance as proxy for road quality
        accelerations = [point.acceleration_ms2 for point in telematics_data]
        if len(accelerations) < 2:
            return 100.0
        
        # Calculate variance
        mean_accel = sum(accelerations) / len(accelerations)
        variance = sum((accel - mean_accel) ** 2 for accel in accelerations) / len(accelerations)
        
        # Convert variance to score (lower variance = better road quality)
        max_variance = 10.0  # Assume max reasonable variance
        quality_score = max(0, 100 - (variance / max_variance) * 100)
        
        return quality_score
    
    def _calculate_average_engine_rpm(self, telematics_data: List[TelematicsDataPoint]) -> float:
        """Calculate average engine RPM during trip."""
        if not telematics_data:
            return 0.0
        
        rpms = [point.engine_rpm for point in telematics_data if point.engine_rpm > 0]
        return sum(rpms) / len(rpms) if rpms else 0.0
    
    def _calculate_fuel_efficiency_score(self, telematics_data: List[TelematicsDataPoint], distance_km: float) -> float:
        """Calculate fuel efficiency score (0-100, higher is better)."""
        if not telematics_data or distance_km == 0:
            return 100.0
        
        # Simple efficiency calculation based on speed and acceleration patterns
        speeds = [point.speed_kmh for point in telematics_data if point.speed_kmh > 0]
        accelerations = [point.acceleration_ms2 for point in telematics_data]
        
        if not speeds:
            return 100.0
        
        average_speed = sum(speeds) / len(speeds)
        
        # Calculate acceleration variance
        if len(accelerations) < 2:
            acceleration_variance = 0
        else:
            mean_accel = sum(accelerations) / len(accelerations)
            acceleration_variance = sum((accel - mean_accel) ** 2 for accel in accelerations) / len(accelerations)
        
        # Efficiency score based on speed and smoothness
        speed_score = min(100, (average_speed / 80) * 100)  # 80 km/h as optimal
        smoothness_score = max(0, 100 - (acceleration_variance / 5) * 100)
        
        # Weighted combination
        efficiency_score = (speed_score * 0.6) + (smoothness_score * 0.4)
        
        return max(0.0, min(100.0, efficiency_score))
