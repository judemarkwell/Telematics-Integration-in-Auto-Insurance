"""
Telematics data simulator for development and testing.

This module simulates real-time telematics data collection from vehicle sensors
and GPS devices. In production, this would be replaced with actual hardware integration.
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional, Dict
import math
from dataclasses import dataclass
from enum import Enum

from ..domain.value_objects import Coordinate, Speed, Acceleration, TimeOfDay
from ..config.settings import config


class RoadType(Enum):
    """Types of roads for driving behavior simulation."""
    HIGHWAY = "highway"
    CITY_STREET = "city_street"
    RESIDENTIAL = "residential"
    RURAL = "rural"
    PARKING_LOT = "parking_lot"


class WeatherCondition(Enum):
    """Weather conditions affecting driving behavior."""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    ICE = "ice"


@dataclass(frozen=True)
class VehicleProfile:
    """Vehicle characteristics affecting driving simulation."""
    make: str
    model: str
    year: int
    engine_type: str  # "gas", "hybrid", "electric"
    safety_rating: float  # 0-5 stars
    theft_probability: float  # 0-1 scale
    age_risk_factor: float  # 0-1 scale
    performance_rating: float  # 0-1 scale (affects acceleration/speed)


@dataclass(frozen=True)
class LocationRiskProfile:
    """Location-based risk factors."""
    crime_rate: float  # 0-1 scale
    accident_frequency: float  # accidents per 1000 vehicles
    road_quality_score: float  # 0-1 scale
    traffic_density: float  # 0-1 scale
    speed_limit: float  # km/h


class TelematicsDataPoint:
    """Represents a single telematics data point with enhanced sensor data."""
    
    def __init__(
        self,
        timestamp: datetime,
        coordinate: Coordinate,
        speed: Speed,
        acceleration: Acceleration,
        heading: float,
        accuracy: float,
        road_type: RoadType = RoadType.CITY_STREET,
        weather: WeatherCondition = WeatherCondition.CLEAR,
        gyroscope_x: float = 0.0,
        gyroscope_y: float = 0.0,
        gyroscope_z: float = 0.0,
        magnetometer_x: float = 0.0,
        magnetometer_y: float = 0.0,
        magnetometer_z: float = 0.0,
        engine_rpm: float = 0.0,
        fuel_level: float = 1.0,
        odometer_reading: float = 0.0
    ):
        self.timestamp = timestamp
        self.coordinate = coordinate
        self.speed = speed
        self.acceleration = acceleration
        self.heading = heading  # Direction in degrees (0-360)
        self.accuracy = accuracy  # GPS accuracy in meters
        self.road_type = road_type
        self.weather = weather
        self.gyroscope_x = gyroscope_x  # Angular velocity around X-axis (rad/s)
        self.gyroscope_y = gyroscope_y  # Angular velocity around Y-axis (rad/s)
        self.gyroscope_z = gyroscope_z  # Angular velocity around Z-axis (rad/s)
        self.magnetometer_x = magnetometer_x  # Magnetic field X component (μT)
        self.magnetometer_y = magnetometer_y  # Magnetic field Y component (μT)
        self.magnetometer_z = magnetometer_z  # Magnetic field Z component (μT)
        self.engine_rpm = engine_rpm  # Engine RPM
        self.fuel_level = fuel_level  # Fuel level (0-1)
        self.odometer_reading = odometer_reading  # Cumulative distance in km


class TelematicsSimulator:
    """
    Enhanced telematics data collection simulator with realistic driving patterns.
    
    This class generates comprehensive telematics data including:
    - GPS coordinates following road-like paths
    - Speed variations based on traffic patterns, road type, and weather
    - Acceleration/deceleration events with vehicle-specific characteristics
    - Time-of-day effects on driving behavior
    - Multi-sensor data (gyroscope, magnetometer, engine data)
    - Weather and road condition impacts
    - Cumulative odometer tracking
    """
    
    def __init__(
        self, 
        driver_id: str, 
        vehicle_id: str,
        vehicle_profile: Optional[VehicleProfile] = None,
        location_profile: Optional[LocationRiskProfile] = None
    ):
        self.driver_id = driver_id
        self.vehicle_id = vehicle_id
        self.sampling_rate = config.telematics.sampling_rate_hz
        self.is_running = False
        self.current_trip_id: Optional[str] = None
        
        # Vehicle and location profiles
        self.vehicle_profile = vehicle_profile or self._get_default_vehicle_profile()
        self.location_profile = location_profile or self._get_default_location_profile()
        
        # Simulation state
        self._current_position = Coordinate(40.7128, -74.0060)  # NYC coordinates
        self._current_speed = Speed(0.0)
        self._current_heading = 0.0
        self._trip_start_time: Optional[datetime] = None
        self._cumulative_distance = 0.0  # Odometer reading
        self._current_road_type = RoadType.CITY_STREET
        self._current_weather = WeatherCondition.CLEAR
        self._previous_position = self._current_position
        
        # Sensor state
        self._gyroscope_x = 0.0
        self._gyroscope_y = 0.0
        self._gyroscope_z = 0.0
        self._engine_rpm = 0.0
        self._fuel_level = 1.0
        
        # Driving behavior modifiers
        self._aggressiveness = random.uniform(0.3, 0.8)  # Driver-specific factor
        self._weather_impact = 1.0  # Weather-based speed/acceleration modifier
    
    def _get_default_vehicle_profile(self) -> VehicleProfile:
        """Get a default vehicle profile for simulation."""
        return VehicleProfile(
            make="Toyota",
            model="Camry",
            year=2020,
            engine_type="gas",
            safety_rating=4.5,
            theft_probability=0.02,
            age_risk_factor=0.1,
            performance_rating=0.6
        )
    
    def _get_default_location_profile(self) -> LocationRiskProfile:
        """Get a default location risk profile for simulation."""
        return LocationRiskProfile(
            crime_rate=0.3,
            accident_frequency=2.5,
            road_quality_score=0.7,
            traffic_density=0.6,
            speed_limit=50.0
        )
    
    def _determine_road_type(self, speed: float, location: Coordinate) -> RoadType:
        """Determine road type based on speed and location patterns."""
        # Simple heuristic based on speed and position
        if speed > 80:
            return RoadType.HIGHWAY
        elif speed > 50:
            return RoadType.CITY_STREET
        elif speed > 20:
            return RoadType.RESIDENTIAL
        elif speed > 5:
            return RoadType.PARKING_LOT
        else:
            return RoadType.RURAL
    
    def _simulate_weather_change(self) -> WeatherCondition:
        """Simulate weather condition changes during trip."""
        # 5% chance of weather change per data point
        if random.random() < 0.05:
            weather_options = list(WeatherCondition)
            return random.choice(weather_options)
        return self._current_weather
    
    def _calculate_weather_impact(self, weather: WeatherCondition) -> float:
        """Calculate speed/acceleration impact based on weather."""
        weather_impacts = {
            WeatherCondition.CLEAR: 1.0,
            WeatherCondition.RAIN: 0.8,
            WeatherCondition.SNOW: 0.6,
            WeatherCondition.FOG: 0.7,
            WeatherCondition.ICE: 0.4
        }
        return weather_impacts.get(weather, 1.0)
        
    async def start_trip(self, start_location: Optional[Coordinate] = None) -> str:
        """Start a new driving trip and return trip ID."""
        if self.is_running:
            raise RuntimeError("Trip is already in progress")
        
        self.current_trip_id = f"trip_{int(time.time())}"
        self._trip_start_time = datetime.utcnow()
        
        if start_location:
            self._current_position = start_location
        
        self.is_running = True
        return self.current_trip_id
    
    async def end_trip(self) -> None:
        """End the current driving trip."""
        if not self.is_running:
            raise RuntimeError("No trip in progress")
        
        self.is_running = False
        self.current_trip_id = None
        self._trip_start_time = None
        self._current_speed = Speed(0.0)
    
    async def collect_data(self) -> AsyncGenerator[TelematicsDataPoint, None]:
        """
        Continuously collect telematics data during a trip.
        
        Yields:
            TelematicsDataPoint: Real-time telematics data
        """
        if not self.is_running:
            raise RuntimeError("No trip in progress")
        
        while self.is_running:
            # Generate realistic driving data
            data_point = self._generate_data_point()
            yield data_point
            
            # Wait for next sampling interval
            await asyncio.sleep(1.0 / self.sampling_rate)
    
    def _generate_data_point(self) -> TelematicsDataPoint:
        """Generate a single realistic telematics data point with enhanced sensor data."""
        now = datetime.utcnow()
        time_of_day = TimeOfDay.from_datetime(now)
        
        # Update weather conditions
        self._current_weather = self._simulate_weather_change()
        self._weather_impact = self._calculate_weather_impact(self._current_weather)
        
        # Simulate driving behavior based on time of day, weather, and vehicle
        base_speed_variance = 0.25
        base_acceleration_variance = 0.3
        
        if time_of_day.is_rush_hour():
            # More aggressive driving during rush hour
            speed_variance = base_speed_variance * 1.2
            acceleration_variance = base_acceleration_variance * 1.3
        elif time_of_day.is_night_time():
            # More conservative driving at night
            speed_variance = base_speed_variance * 0.8
            acceleration_variance = base_acceleration_variance * 0.7
        else:
            # Normal driving conditions
            speed_variance = base_speed_variance
            acceleration_variance = base_acceleration_variance
        
        # Apply weather impact
        speed_variance *= self._weather_impact
        acceleration_variance *= self._weather_impact
        
        # Apply vehicle performance characteristics
        speed_variance *= (1.0 + self.vehicle_profile.performance_rating * 0.3)
        acceleration_variance *= (1.0 + self.vehicle_profile.performance_rating * 0.4)
        
        # Update position (simulate movement along a path)
        self._update_position()
        
        # Update speed with realistic variations
        self._update_speed(speed_variance)
        
        # Generate acceleration based on speed changes
        acceleration = self._calculate_acceleration()
        
        # Generate heading (direction of movement)
        self._update_heading()
        
        # Determine road type based on current conditions
        self._current_road_type = self._determine_road_type(
            self._current_speed.value, 
            self._current_position
        )
        
        # Generate sensor data
        gyro_data = self._generate_gyroscope_data()
        magnetometer_data = self._generate_magnetometer_data()
        engine_data = self._generate_engine_data()
        
        # Update cumulative distance (odometer)
        if self._current_speed.value > 0:
            distance_increment = self._previous_position.distance_to(self._current_position)
            self._cumulative_distance += distance_increment
        
        # GPS accuracy varies (typically 3-8 meters, worse in bad weather)
        base_accuracy = random.uniform(3.0, 8.0)
        if self._current_weather in [WeatherCondition.RAIN, WeatherCondition.SNOW, WeatherCondition.FOG]:
            base_accuracy *= 1.5  # Worse GPS accuracy in bad weather
        
        accuracy = base_accuracy
        
        # Update previous position for next iteration
        self._previous_position = self._current_position
        
        return TelematicsDataPoint(
            timestamp=now,
            coordinate=self._current_position,
            speed=self._current_speed,
            acceleration=acceleration,
            heading=self._current_heading,
            accuracy=accuracy,
            road_type=self._current_road_type,
            weather=self._current_weather,
            gyroscope_x=gyro_data[0],
            gyroscope_y=gyro_data[1],
            gyroscope_z=gyro_data[2],
            magnetometer_x=magnetometer_data[0],
            magnetometer_y=magnetometer_data[1],
            magnetometer_z=magnetometer_data[2],
            engine_rpm=engine_data,
            fuel_level=self._fuel_level,
            odometer_reading=self._cumulative_distance
        )
    
    def _update_position(self) -> None:
        """Update GPS position based on current speed and heading."""
        if self._current_speed.value == 0:
            return
        
        # Convert speed from km/h to degrees per second (rough approximation)
        # 1 degree latitude ≈ 111 km
        speed_deg_per_sec = self._current_speed.value / (111 * 3600)
        
        # Calculate new position
        lat_delta = speed_deg_per_sec * math.cos(math.radians(self._current_heading))
        lon_delta = speed_deg_per_sec * math.sin(math.radians(self._current_heading))
        
        new_lat = self._current_position.latitude + lat_delta
        new_lon = self._current_position.longitude + lon_delta
        
        # Add some realistic GPS noise
        noise_lat = random.uniform(-0.0001, 0.0001)
        noise_lon = random.uniform(-0.0001, 0.0001)
        
        self._current_position = Coordinate(
            latitude=new_lat + noise_lat,
            longitude=new_lon + noise_lon
        )
    
    def _update_speed(self, variance: float) -> None:
        """Update speed with realistic driving patterns."""
        # Simulate different driving scenarios
        scenario = random.random()
        
        if scenario < 0.1:  # 10% chance of stopping
            target_speed = 0.0
        elif scenario < 0.3:  # 20% chance of slow speed (city driving)
            target_speed = random.uniform(20, 40)
        elif scenario < 0.7:  # 40% chance of moderate speed
            target_speed = random.uniform(40, 70)
        elif scenario < 0.9:  # 20% chance of highway speed
            target_speed = random.uniform(70, 100)
        else:  # 10% chance of high speed
            target_speed = random.uniform(100, 120)
        
        # Smooth speed transitions
        speed_diff = target_speed - self._current_speed.value
        speed_change = speed_diff * variance * random.uniform(0.1, 0.3)
        
        new_speed = max(0, self._current_speed.value + speed_change)
        self._current_speed = Speed(new_speed)
    
    def _calculate_acceleration(self) -> Acceleration:
        """Calculate acceleration based on speed changes."""
        # This is a simplified calculation - in reality, acceleration
        # would be measured directly by accelerometer sensors
        if not hasattr(self, '_previous_speed'):
            self._previous_speed = self._current_speed.value
            return Acceleration(0.0)
        
        speed_diff = self._current_speed.value - self._previous_speed
        # Convert from km/h to m/s² (rough approximation)
        acceleration_value = speed_diff * 0.278  # 1 km/h ≈ 0.278 m/s
        
        # Add some realistic noise
        noise = random.uniform(-0.5, 0.5)
        acceleration_value += noise
        
        self._previous_speed = self._current_speed.value
        return Acceleration(acceleration_value)
    
    def _update_heading(self) -> None:
        """Update heading (direction) with realistic turns."""
        if self._current_speed.value < 5:  # Don't change direction when stopped
            return
        
        # Simulate gradual turns
        turn_probability = 0.1  # 10% chance of turning each update
        if random.random() < turn_probability:
            turn_angle = random.uniform(-30, 30)  # Turn up to 30 degrees
            self._current_heading = (self._current_heading + turn_angle) % 360
    
    def _generate_gyroscope_data(self) -> tuple[float, float, float]:
        """Generate realistic gyroscope data based on vehicle movement."""
        # Gyroscope measures angular velocity in rad/s
        # X-axis: pitch (forward/backward rotation)
        # Y-axis: roll (left/right rotation) 
        # Z-axis: yaw (rotation around vertical axis)
        
        # Base noise level
        noise_level = 0.1
        
        # X-axis (pitch) - affected by acceleration/deceleration
        pitch = 0.0
        if abs(self._current_speed.value - (getattr(self, '_previous_speed', 0))) > 5:
            pitch = random.uniform(-0.5, 0.5) * self._aggressiveness
        
        # Y-axis (roll) - affected by turns and road banking
        roll = 0.0
        if hasattr(self, '_previous_heading'):
            heading_change = abs(self._current_heading - self._previous_heading)
            if heading_change > 5:  # Significant turn
                roll = random.uniform(-0.3, 0.3) * (heading_change / 30.0)
        
        # Z-axis (yaw) - directly related to heading changes
        yaw = 0.0
        if hasattr(self, '_previous_heading'):
            heading_change = self._current_heading - self._previous_heading
            if heading_change > 180:
                heading_change -= 360
            elif heading_change < -180:
                heading_change += 360
            yaw = math.radians(heading_change) * 10  # Convert to rad/s
        
        # Add noise
        gyro_x = pitch + random.uniform(-noise_level, noise_level)
        gyro_y = roll + random.uniform(-noise_level, noise_level)
        gyro_z = yaw + random.uniform(-noise_level, noise_level)
        
        # Store current values for next iteration
        self._previous_heading = self._current_heading
        
        return (gyro_x, gyro_y, gyro_z)
    
    def _generate_magnetometer_data(self) -> tuple[float, float, float]:
        """Generate realistic magnetometer data (compass readings)."""
        # Magnetometer measures magnetic field in microtesla (μT)
        # Typical Earth's magnetic field: ~25-65 μT
        
        # Base magnetic field strength
        base_field = 45.0
        
        # X-axis: typically points north
        mag_x = base_field * math.cos(math.radians(self._current_heading))
        
        # Y-axis: typically points east  
        mag_y = base_field * math.sin(math.radians(self._current_heading))
        
        # Z-axis: typically points down (vertical component)
        mag_z = -base_field * 0.5  # Vertical component is typically smaller
        
        # Add noise and local magnetic anomalies
        noise = 2.0
        mag_x += random.uniform(-noise, noise)
        mag_y += random.uniform(-noise, noise)
        mag_z += random.uniform(-noise, noise)
        
        return (mag_x, mag_y, mag_z)
    
    def _generate_engine_data(self) -> float:
        """Generate realistic engine RPM data."""
        # Engine RPM typically ranges from 600-6000 RPM
        # Idle: ~600-1000 RPM
        # Cruising: ~1500-3000 RPM
        # Acceleration: ~3000-6000 RPM
        
        if self._current_speed.value == 0:
            # Idle RPM
            self._engine_rpm = random.uniform(600, 1000)
        elif self._current_speed.value < 30:
            # Low speed driving
            self._engine_rpm = random.uniform(1000, 2000)
        elif self._current_speed.value < 60:
            # City driving
            self._engine_rpm = random.uniform(1500, 2500)
        elif self._current_speed.value < 100:
            # Highway cruising
            self._engine_rpm = random.uniform(2000, 3000)
        else:
            # High speed driving
            self._engine_rpm = random.uniform(2500, 4000)
        
        # Adjust for vehicle performance characteristics
        performance_modifier = 1.0 + (self.vehicle_profile.performance_rating - 0.5) * 0.3
        self._engine_rpm *= performance_modifier
        
        # Add some noise
        self._engine_rpm += random.uniform(-50, 50)
        
        # Ensure realistic bounds
        self._engine_rpm = max(600, min(6000, self._engine_rpm))
        
        return self._engine_rpm


class TelematicsDataCollector:
    """
    High-level interface for telematics data collection.
    
    This class manages the lifecycle of data collection sessions and
    provides a clean API for the rest of the application.
    """
    
    def __init__(self):
        self.active_simulators: dict[str, TelematicsSimulator] = {}
        self.vehicle_profiles: dict[str, VehicleProfile] = {}
        self.location_profiles: dict[str, LocationRiskProfile] = {}
    
    def register_vehicle_profile(self, vehicle_id: str, profile: VehicleProfile) -> None:
        """Register a vehicle profile for enhanced simulation."""
        self.vehicle_profiles[vehicle_id] = profile
    
    def register_location_profile(self, location_id: str, profile: LocationRiskProfile) -> None:
        """Register a location risk profile for enhanced simulation."""
        self.location_profiles[location_id] = profile
    
    async def start_collection(
        self, 
        driver_id: str, 
        vehicle_id: str,
        start_location: Optional[Coordinate] = None,
        location_id: Optional[str] = None
    ) -> str:
        """Start telematics data collection for a driver/vehicle."""
        # Get vehicle and location profiles if available
        vehicle_profile = self.vehicle_profiles.get(vehicle_id)
        location_profile = self.location_profiles.get(location_id) if location_id else None
        
        simulator = TelematicsSimulator(
            driver_id, 
            vehicle_id, 
            vehicle_profile=vehicle_profile,
            location_profile=location_profile
        )
        trip_id = await simulator.start_trip(start_location)
        
        self.active_simulators[trip_id] = simulator
        return trip_id
    
    async def stop_collection(self, trip_id: str) -> None:
        """Stop telematics data collection for a trip."""
        if trip_id not in self.active_simulators:
            raise ValueError(f"No active collection for trip {trip_id}")
        
        simulator = self.active_simulators[trip_id]
        await simulator.end_trip()
        del self.active_simulators[trip_id]
    
    async def get_data_stream(self, trip_id: str) -> AsyncGenerator[TelematicsDataPoint, None]:
        """Get real-time data stream for a trip."""
        if trip_id not in self.active_simulators:
            raise ValueError(f"No active collection for trip {trip_id}")
        
        simulator = self.active_simulators[trip_id]
        async for data_point in simulator.collect_data():
            yield data_point
    
    def get_active_trips(self) -> List[str]:
        """Get list of currently active trip IDs."""
        return list(self.active_simulators.keys())
