"""
Data processing engine for telematics data.

This module handles real-time processing, cleaning, and analysis of telematics data
to extract meaningful driving behavior patterns and metrics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from ..data_collection.telematics_simulator import TelematicsDataPoint, RoadType, WeatherCondition
from ..domain.value_objects import Coordinate, Speed, Acceleration, Distance, TimeOfDay


@dataclass(frozen=True)
class DrivingEvent:
    """Represents a significant driving event."""
    event_type: str  # 'hard_brake', 'hard_acceleration', 'speeding', 'sharp_turn'
    timestamp: datetime
    severity: float  # 0-1 scale
    location: Coordinate
    details: Dict[str, float]


@dataclass(frozen=True)
class TripMetrics:
    """Aggregated metrics for a driving trip with enhanced telematics data."""
    trip_id: str
    driver_id: str
    start_time: datetime
    end_time: datetime
    total_distance: Distance
    total_duration: timedelta
    average_speed: Speed
    max_speed: Speed
    hard_braking_events: int
    hard_acceleration_events: int
    speeding_events: int
    sharp_turn_events: int
    night_driving_percentage: float
    rush_hour_percentage: float
    overall_score: float  # 0-100 scale
    # Enhanced metrics
    highway_driving_percentage: float
    city_driving_percentage: float
    residential_driving_percentage: float
    weather_impact_score: float
    road_quality_score: float
    average_engine_rpm: float
    fuel_efficiency_score: float
    odometer_reading: float


class TelematicsDataProcessor:
    """
    Processes raw telematics data to extract driving behavior insights.
    
    This class implements real-time data processing algorithms to:
    - Clean and validate incoming data
    - Detect driving events (hard braking, acceleration, etc.)
    - Calculate trip-level metrics
    - Generate risk indicators
    """
    
    def __init__(self):
        self.active_trips: Dict[str, List[TelematicsDataPoint]] = {}
        self.trip_events: Dict[str, List[DrivingEvent]] = {}
        
        # Event detection thresholds
        self.hard_brake_threshold = -3.0  # m/s²
        self.hard_acceleration_threshold = 3.0  # m/s²
        self.speeding_threshold = 80.0  # km/h
        self.sharp_turn_threshold = 45.0  # degrees/second
        
    async def process_data_point(
        self, 
        trip_id: str, 
        data_point: TelematicsDataPoint
    ) -> List[DrivingEvent]:
        """
        Process a single telematics data point and return any detected events.
        
        Args:
            trip_id: Unique identifier for the trip
            data_point: Raw telematics data point
            
        Returns:
            List of detected driving events
        """
        # Store data point
        if trip_id not in self.active_trips:
            self.active_trips[trip_id] = []
            self.trip_events[trip_id] = []
        
        self.active_trips[trip_id].append(data_point)
        
        # Detect events
        events = self._detect_events(trip_id, data_point)
        
        # Store events
        self.trip_events[trip_id].extend(events)
        
        return events
    
    def _detect_events(
        self, 
        trip_id: str, 
        data_point: TelematicsDataPoint
    ) -> List[DrivingEvent]:
        """Detect driving events from a data point."""
        events = []
        
        # Hard braking detection
        if data_point.acceleration.is_hard_braking(self.hard_brake_threshold):
            severity = min(1.0, abs(data_point.acceleration.value) / 5.0)
            event = DrivingEvent(
                event_type="hard_brake",
                timestamp=data_point.timestamp,
                severity=severity,
                location=data_point.coordinate,
                details={
                    "acceleration": data_point.acceleration.value,
                    "speed": data_point.speed.value
                }
            )
            events.append(event)
        
        # Hard acceleration detection
        if data_point.acceleration.is_hard_acceleration(self.hard_acceleration_threshold):
            severity = min(1.0, data_point.acceleration.value / 5.0)
            event = DrivingEvent(
                event_type="hard_acceleration",
                timestamp=data_point.timestamp,
                severity=severity,
                location=data_point.coordinate,
                details={
                    "acceleration": data_point.acceleration.value,
                    "speed": data_point.speed.value
                }
            )
            events.append(event)
        
        # Speeding detection
        if data_point.speed.is_speeding(self.speeding_threshold):
            severity = min(1.0, (data_point.speed.value - self.speeding_threshold) / 50.0)
            event = DrivingEvent(
                event_type="speeding",
                timestamp=data_point.timestamp,
                severity=severity,
                location=data_point.coordinate,
                details={
                    "speed": data_point.speed.value,
                    "speed_limit": self.speeding_threshold
                }
            )
            events.append(event)
        
        # Sharp turn detection (requires previous data point)
        if len(self.active_trips[trip_id]) > 1:
            sharp_turn_event = self._detect_sharp_turn(trip_id, data_point)
            if sharp_turn_event:
                events.append(sharp_turn_event)
        
        return events
    
    def _detect_sharp_turn(
        self, 
        trip_id: str, 
        current_point: TelematicsDataPoint
    ) -> Optional[DrivingEvent]:
        """Detect sharp turns by analyzing heading changes."""
        data_points = self.active_trips[trip_id]
        if len(data_points) < 2:
            return None
        
        previous_point = data_points[-2]
        
        # Calculate heading change
        heading_change = abs(current_point.heading - previous_point.heading)
        if heading_change > 180:
            heading_change = 360 - heading_change
        
        # Calculate time difference
        time_diff = (current_point.timestamp - previous_point.timestamp).total_seconds()
        if time_diff == 0:
            return None
        
        # Calculate angular velocity (degrees per second)
        angular_velocity = heading_change / time_diff
        
        if angular_velocity > self.sharp_turn_threshold:
            severity = min(1.0, angular_velocity / 90.0)
            return DrivingEvent(
                event_type="sharp_turn",
                timestamp=current_point.timestamp,
                severity=severity,
                location=current_point.coordinate,
                details={
                    "angular_velocity": angular_velocity,
                    "heading_change": heading_change,
                    "speed": current_point.speed.value
                }
            )
        
        return None
    
    def calculate_trip_metrics(self, trip_id: str) -> Optional[TripMetrics]:
        """
        Calculate comprehensive metrics for a completed trip.
        
        Args:
            trip_id: Unique identifier for the trip
            
        Returns:
            TripMetrics object with aggregated trip data
        """
        if trip_id not in self.active_trips or not self.active_trips[trip_id]:
            return None
        
        data_points = self.active_trips[trip_id]
        events = self.trip_events.get(trip_id, [])
        
        # Basic trip information
        start_time = data_points[0].timestamp
        end_time = data_points[-1].timestamp
        duration = end_time - start_time
        
        # Calculate total distance
        total_distance = self._calculate_total_distance(data_points)
        
        # Calculate speed metrics
        speeds = [dp.speed.value for dp in data_points if dp.speed.value > 0]
        average_speed = Speed(statistics.mean(speeds)) if speeds else Speed(0.0)
        max_speed = Speed(max(speeds)) if speeds else Speed(0.0)
        
        # Count events by type
        event_counts = {
            "hard_braking": len([e for e in events if e.event_type == "hard_brake"]),
            "hard_acceleration": len([e for e in events if e.event_type == "hard_acceleration"]),
            "speeding": len([e for e in events if e.event_type == "speeding"]),
            "sharp_turn": len([e for e in events if e.event_type == "sharp_turn"])
        }
        
        # Calculate time-based metrics
        night_driving_pct = self._calculate_night_driving_percentage(data_points)
        rush_hour_pct = self._calculate_rush_hour_percentage(data_points)
        
        # Calculate enhanced metrics
        road_type_stats = self._calculate_road_type_percentages(data_points)
        weather_impact = self._calculate_weather_impact_score(data_points)
        road_quality = self._calculate_road_quality_score(data_points)
        engine_stats = self._calculate_engine_metrics(data_points)
        fuel_efficiency = self._calculate_fuel_efficiency_score(data_points, total_distance)
        odometer_reading = data_points[-1].odometer_reading if data_points else 0.0
        
        # Calculate overall score (0-100, higher is better)
        overall_score = self._calculate_overall_score(
            event_counts, average_speed, max_speed, weather_impact, road_quality
        )
        
        # Extract driver_id from first data point (would be set in real implementation)
        driver_id = "driver_001"  # This would come from the data collection system
        
        return TripMetrics(
            trip_id=trip_id,
            driver_id=driver_id,
            start_time=start_time,
            end_time=end_time,
            total_distance=total_distance,
            total_duration=duration,
            average_speed=average_speed,
            max_speed=max_speed,
            hard_braking_events=event_counts["hard_braking"],
            hard_acceleration_events=event_counts["hard_acceleration"],
            speeding_events=event_counts["speeding"],
            sharp_turn_events=event_counts["sharp_turn"],
            night_driving_percentage=night_driving_pct,
            rush_hour_percentage=rush_hour_pct,
            overall_score=overall_score,
            highway_driving_percentage=road_type_stats["highway"],
            city_driving_percentage=road_type_stats["city"],
            residential_driving_percentage=road_type_stats["residential"],
            weather_impact_score=weather_impact,
            road_quality_score=road_quality,
            average_engine_rpm=engine_stats["average_rpm"],
            fuel_efficiency_score=fuel_efficiency,
            odometer_reading=odometer_reading
        )
    
    def _calculate_total_distance(self, data_points: List[TelematicsDataPoint]) -> Distance:
        """Calculate total distance traveled during the trip."""
        if len(data_points) < 2:
            return Distance(0.0)
        
        total_distance = 0.0
        for i in range(1, len(data_points)):
            prev_point = data_points[i-1]
            curr_point = data_points[i]
            distance = prev_point.coordinate.distance_to(curr_point.coordinate)
            total_distance += distance
        
        return Distance(total_distance)
    
    def _calculate_night_driving_percentage(self, data_points: List[TelematicsDataPoint]) -> float:
        """Calculate percentage of trip that occurred during night time."""
        if not data_points:
            return 0.0
        
        night_points = 0
        for point in data_points:
            time_of_day = TimeOfDay.from_datetime(point.timestamp)
            if time_of_day.is_night_time():
                night_points += 1
        
        return (night_points / len(data_points)) * 100
    
    def _calculate_rush_hour_percentage(self, data_points: List[TelematicsDataPoint]) -> float:
        """Calculate percentage of trip that occurred during rush hour."""
        if not data_points:
            return 0.0
        
        rush_hour_points = 0
        for point in data_points:
            time_of_day = TimeOfDay.from_datetime(point.timestamp)
            if time_of_day.is_rush_hour():
                rush_hour_points += 1
        
        return (rush_hour_points / len(data_points)) * 100
    
    def _calculate_road_type_percentages(self, data_points: List[TelematicsDataPoint]) -> Dict[str, float]:
        """Calculate percentage of trip spent on different road types."""
        if not data_points:
            return {"highway": 0.0, "city": 0.0, "residential": 0.0}
        
        road_type_counts = {
            RoadType.HIGHWAY: 0,
            RoadType.CITY_STREET: 0,
            RoadType.RESIDENTIAL: 0,
            RoadType.RURAL: 0,
            RoadType.PARKING_LOT: 0
        }
        
        for point in data_points:
            road_type_counts[point.road_type] += 1
        
        total_points = len(data_points)
        return {
            "highway": (road_type_counts[RoadType.HIGHWAY] / total_points) * 100,
            "city": (road_type_counts[RoadType.CITY_STREET] / total_points) * 100,
            "residential": (road_type_counts[RoadType.RESIDENTIAL] / total_points) * 100
        }
    
    def _calculate_weather_impact_score(self, data_points: List[TelematicsDataPoint]) -> float:
        """Calculate weather impact score (0-100, higher is better)."""
        if not data_points:
            return 100.0
        
        weather_impacts = {
            WeatherCondition.CLEAR: 100.0,
            WeatherCondition.RAIN: 80.0,
            WeatherCondition.SNOW: 60.0,
            WeatherCondition.FOG: 70.0,
            WeatherCondition.ICE: 40.0
        }
        
        total_score = 0.0
        for point in data_points:
            total_score += weather_impacts.get(point.weather, 100.0)
        
        return total_score / len(data_points)
    
    def _calculate_road_quality_score(self, data_points: List[TelematicsDataPoint]) -> float:
        """Calculate road quality score based on driving smoothness."""
        if len(data_points) < 2:
            return 100.0
        
        # Calculate smoothness based on acceleration variance
        accelerations = [point.acceleration.value for point in data_points]
        acceleration_variance = statistics.variance(accelerations) if len(accelerations) > 1 else 0
        
        # Lower variance = smoother driving = better road quality
        # Convert variance to score (0-100)
        max_variance = 10.0  # Assume max reasonable variance
        quality_score = max(0, 100 - (acceleration_variance / max_variance) * 100)
        
        return quality_score
    
    def _calculate_engine_metrics(self, data_points: List[TelematicsDataPoint]) -> Dict[str, float]:
        """Calculate engine-related metrics."""
        if not data_points:
            return {"average_rpm": 0.0}
        
        rpms = [point.engine_rpm for point in data_points if point.engine_rpm > 0]
        average_rpm = statistics.mean(rpms) if rpms else 0.0
        
        return {"average_rpm": average_rpm}
    
    def _calculate_fuel_efficiency_score(self, data_points: List[TelematicsDataPoint], total_distance: Distance) -> float:
        """Calculate fuel efficiency score (0-100, higher is better)."""
        if not data_points or total_distance.value == 0:
            return 100.0
        
        # Simple fuel efficiency calculation based on speed and acceleration patterns
        # Higher average speed with lower acceleration variance = better efficiency
        
        speeds = [point.speed.value for point in data_points if point.speed.value > 0]
        accelerations = [point.acceleration.value for point in data_points]
        
        if not speeds:
            return 100.0
        
        average_speed = statistics.mean(speeds)
        acceleration_variance = statistics.variance(accelerations) if len(accelerations) > 1 else 0
        
        # Efficiency score based on speed (higher is better) and acceleration smoothness
        speed_score = min(100, (average_speed / 80) * 100)  # 80 km/h as optimal
        smoothness_score = max(0, 100 - (acceleration_variance / 5) * 100)  # Lower variance is better
        
        # Weighted combination
        efficiency_score = (speed_score * 0.6) + (smoothness_score * 0.4)
        
        return max(0.0, min(100.0, efficiency_score))
    
    def _calculate_overall_score(
        self, 
        event_counts: Dict[str, int], 
        average_speed: Speed, 
        max_speed: Speed,
        weather_impact: float,
        road_quality: float
    ) -> float:
        """
        Calculate overall driving score (0-100, higher is better).
        
        Enhanced scoring algorithm that considers weather and road conditions.
        """
        base_score = 100.0
        
        # Penalize events
        event_penalties = {
            "hard_braking": 5.0,
            "hard_acceleration": 3.0,
            "speeding": 2.0,
            "sharp_turn": 1.0
        }
        
        for event_type, count in event_counts.items():
            penalty = event_penalties.get(event_type, 0.0)
            base_score -= count * penalty
        
        # Penalize excessive speeds
        if max_speed.value > 120:
            base_score -= (max_speed.value - 120) * 0.5
        
        # Bonus for moderate average speeds (40-60 km/h is ideal)
        if 40 <= average_speed.value <= 60:
            base_score += 5.0
        
        # Apply weather and road quality adjustments
        weather_factor = weather_impact / 100.0
        road_quality_factor = road_quality / 100.0
        
        # Adjust score based on conditions (driver should be penalized less in bad conditions)
        adjusted_score = base_score * (0.7 + 0.3 * weather_factor * road_quality_factor)
        
        return max(0.0, min(100.0, adjusted_score))
    
    def get_trip_summary(self, trip_id: str) -> Optional[Dict]:
        """Get a summary of trip data for quick access."""
        if trip_id not in self.active_trips:
            return None
        
        data_points = self.active_trips[trip_id]
        events = self.trip_events.get(trip_id, [])
        
        return {
            "trip_id": trip_id,
            "data_points_count": len(data_points),
            "events_count": len(events),
            "duration_minutes": (data_points[-1].timestamp - data_points[0].timestamp).total_seconds() / 60,
            "last_updated": data_points[-1].timestamp if data_points else None
        }
    
    def clear_trip_data(self, trip_id: str) -> None:
        """Clear all data for a completed trip."""
        if trip_id in self.active_trips:
            del self.active_trips[trip_id]
        if trip_id in self.trip_events:
            del self.trip_events[trip_id]

