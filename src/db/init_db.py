"""
Database initialization script for the Telematics Insurance System.

This script initializes the database with sample data for development and testing.
"""

import asyncio
import random
from datetime import datetime, timedelta
from .engine import init_db, get_session_context
from .repositories import (
    DriverRepository, VehicleRepository, PolicyRepository, 
    LocationRiskProfileRepository, VehicleRiskProfileRepository,
    TripRepository, TelematicsDataPointRepository, DrivingEventRepository,
    RiskScoreRepository, FeatureRowRepository
)
from .models import (
    Driver, Vehicle, Policy, LocationRiskProfile, VehicleRiskProfile,
    Trip, TelematicsDataPoint, DrivingEvent, RiskScore, FeatureRow
)
from ..data_collection.telematics_simulator import (
    TelematicsDataCollector, TelematicsSimulator, TelematicsDataPoint as SimDataPoint,
    RoadType, WeatherCondition, VehicleProfile, LocationRiskProfile as SimLocationProfile
)
from ..data_processing.data_processor import TelematicsDataProcessor
from ..data_processing.features import FeatureExtractor
from ..risk_scoring.risk_scorer import RiskScoringService


def create_sample_data():
    """Create sample data for development and testing."""
    print("🌱 Creating sample data...")
    
    with get_session_context() as session:
        # Create repositories
        driver_repo = DriverRepository()
        vehicle_repo = VehicleRepository()
        policy_repo = PolicyRepository()
        location_repo = LocationRiskProfileRepository()
        vehicle_profile_repo = VehicleRiskProfileRepository()
        
        # Create sample location profiles
        locations = [
            LocationRiskProfile(
                location_name="Downtown NYC",
                latitude=40.7128,
                longitude=-74.0060,
                crime_rate=0.6,
                accident_frequency=3.5,
                road_quality_score=0.6,
                traffic_density=0.9,
                speed_limit_kmh=35.0
            ),
            LocationRiskProfile(
                location_name="Suburban NJ",
                latitude=40.7589,
                longitude=-74.0424,
                crime_rate=0.2,
                accident_frequency=1.8,
                road_quality_score=0.8,
                traffic_density=0.4,
                speed_limit_kmh=50.0
            ),
            LocationRiskProfile(
                location_name="Highway I-95",
                latitude=40.6892,
                longitude=-74.0445,
                crime_rate=0.1,
                accident_frequency=2.2,
                road_quality_score=0.9,
                traffic_density=0.7,
                speed_limit_kmh=100.0
            )
        ]
        
        for location in locations:
            location_repo.create(session, **location.dict())
        
        # Create sample vehicle risk profiles
        vehicle_profiles = [
            VehicleRiskProfile(
                make="Toyota",
                model="Camry",
                year=2020,
                engine_type="gas",
                safety_rating=4.5,
                theft_probability=0.02,
                age_risk_factor=0.1,
                performance_rating=0.6
            ),
            VehicleRiskProfile(
                make="Honda",
                model="Civic",
                year=2019,
                engine_type="gas",
                safety_rating=4.4,
                theft_probability=0.04,
                age_risk_factor=0.15,
                performance_rating=0.6
            ),
            VehicleRiskProfile(
                make="Tesla",
                model="Model 3",
                year=2022,
                engine_type="electric",
                safety_rating=4.8,
                theft_probability=0.01,
                age_risk_factor=0.05,
                performance_rating=0.9
            ),
            VehicleRiskProfile(
                make="BMW",
                model="3 Series",
                year=2021,
                engine_type="gas",
                safety_rating=4.6,
                theft_probability=0.06,
                age_risk_factor=0.08,
                performance_rating=0.8
            )
        ]
        
        for profile in vehicle_profiles:
            vehicle_profile_repo.create(session, **profile.dict())
        
        # Create sample drivers
        drivers = [
            Driver(
                name="John Smith",
                email="john.smith@example.com",
                phone="+1-555-0101",
                license_number="DL123456789",
                date_of_birth=datetime(1985, 5, 15)
            ),
            Driver(
                name="Sarah Johnson",
                email="sarah.johnson@example.com",
                phone="+1-555-0102",
                license_number="DL987654321",
                date_of_birth=datetime(1990, 8, 22)
            ),
            Driver(
                name="Mike Chen",
                email="mike.chen@example.com",
                phone="+1-555-0103",
                license_number="DL456789123",
                date_of_birth=datetime(1988, 12, 3)
            )
        ]
        
        created_drivers = []
        for driver in drivers:
            created_driver = driver_repo.create(session, **driver.dict())
            created_drivers.append(created_driver)
        
        # Create sample vehicles
        vehicles = [
            Vehicle(
                driver_id=created_drivers[0].id,
                make="Toyota",
                model="Camry",
                year=2020,
                vin="1HGBH41JXMN109186",
                license_plate="ABC123",
                engine_type="gas",
                safety_rating=4.5,
                theft_probability=0.02
            ),
            Vehicle(
                driver_id=created_drivers[1].id,
                make="Honda",
                model="Civic",
                year=2019,
                vin="2HGBH41JXMN109187",
                license_plate="XYZ789",
                engine_type="gas",
                safety_rating=4.4,
                theft_probability=0.04
            ),
            Vehicle(
                driver_id=created_drivers[2].id,
                make="Tesla",
                model="Model 3",
                year=2022,
                vin="3HGBH41JXMN109188",
                license_plate="EV2022",
                engine_type="electric",
                safety_rating=4.8,
                theft_probability=0.01
            )
        ]
        
        created_vehicles = []
        for vehicle in vehicles:
            created_vehicle = vehicle_repo.create(session, **vehicle.dict())
            created_vehicles.append(created_vehicle)
        
        # Create sample policies
        policies = [
            Policy(
                driver_id=created_drivers[0].id,
                vehicle_id=created_vehicles[0].id,
                policy_number="POL-001-2024",
                start_date=datetime.utcnow() - timedelta(days=30),
                end_date=datetime.utcnow() + timedelta(days=335),
                base_premium=1200.0,
                current_premium=1200.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[1].id,
                vehicle_id=created_vehicles[1].id,
                policy_number="POL-002-2024",
                start_date=datetime.utcnow() - timedelta(days=15),
                end_date=datetime.utcnow() + timedelta(days=350),
                base_premium=1000.0,
                current_premium=1000.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[2].id,
                vehicle_id=created_vehicles[2].id,
                policy_number="POL-003-2024",
                start_date=datetime.utcnow() - timedelta(days=7),
                end_date=datetime.utcnow() + timedelta(days=358),
                base_premium=1500.0,
                current_premium=1500.0,
                status="active"
            )
        ]
        
        for policy in policies:
            policy_repo.create(session, **policy.dict())
        
        print(f"✅ Created {len(drivers)} drivers, {len(vehicles)} vehicles, {len(policies)} policies")
        print(f"✅ Created {len(locations)} location profiles, {len(vehicle_profiles)} vehicle profiles")
        
        return created_drivers, created_vehicles, created_policies


async def create_simulation_data_for_driver(driver_license_number: str, num_trips: int = 5):
    """Create realistic simulation data for a specific driver."""
    print(f"🎮 Creating simulation data for driver: {driver_license_number}")
    
    with get_session_context() as session:
        # Get repositories
        driver_repo = DriverRepository()
        vehicle_repo = VehicleRepository()
        trip_repo = TripRepository()
        telematics_repo = TelematicsDataPointRepository()
        event_repo = DrivingEventRepository()
        risk_repo = RiskScoreRepository()
        feature_repo = FeatureRowRepository()
        
        # Find driver by license number
        driver = driver_repo.get_by_license(session, driver_license_number)
        if not driver:
            print(f"❌ Driver with license {driver_license_number} not found")
            return None
        
        # Get driver's vehicle
        vehicles = vehicle_repo.get_by_driver(session, driver.id)
        if not vehicles:
            print(f"❌ No vehicle found for driver {driver_license_number}")
            return None
        
        vehicle = vehicles[0]  # Use first vehicle
        
        # Create vehicle and location profiles for simulation
        vehicle_profile = VehicleProfile(
            make=vehicle.make,
            model=vehicle.model,
            year=vehicle.year,
            engine_type=vehicle.engine_type,
            safety_rating=vehicle.safety_rating,
            theft_probability=vehicle.theft_probability,
            age_risk_factor=0.1,  # Default
            performance_rating=0.6  # Default
        )
        
        location_profile = SimLocationProfile(
            crime_rate=0.3,
            accident_frequency=2.5,
            road_quality_score=0.7,
            traffic_density=0.6,
            speed_limit=50.0
        )
        
        # Initialize services
        data_collector = TelematicsDataCollector()
        data_processor = TelematicsDataProcessor()
        risk_scorer = RiskScoringService()
        feature_extractor = FeatureExtractor()
        
        # Register profiles
        data_collector.register_vehicle_profile(str(vehicle.id), vehicle_profile)
        data_collector.register_location_profile("default", location_profile)
        
        all_trip_metrics = []
        
        # Generate multiple trips
        for trip_num in range(num_trips):
            print(f"  🚗 Generating trip {trip_num + 1}/{num_trips}...")
            
            # Create trip in database
            trip = Trip(
                driver_id=driver.id,
                vehicle_id=vehicle.id,
                start_time=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                start_latitude=40.7128 + random.uniform(-0.1, 0.1),
                start_longitude=-74.0060 + random.uniform(-0.1, 0.1)
            )
            created_trip = trip_repo.create(session, **trip.dict())
            
            # Simulate telematics data collection
            start_location = SimDataPoint.Coordinate(
                trip.start_latitude, trip.start_longitude
            )
            
            trip_id = await data_collector.start_collection(
                str(driver.id), str(vehicle.id), start_location, "default"
            )
            
            # Collect data points
            data_points = []
            events = []
            data_count = 0
            
            async for sim_data_point in data_collector.get_data_stream(trip_id):
                # Convert simulation data point to database model
                db_data_point = TelematicsDataPoint(
                    trip_id=created_trip.id,
                    timestamp=sim_data_point.timestamp,
                    latitude=sim_data_point.coordinate.latitude,
                    longitude=sim_data_point.coordinate.longitude,
                    speed_kmh=sim_data_point.speed.value,
                    acceleration_ms2=sim_data_point.acceleration.value,
                    heading_degrees=sim_data_point.heading,
                    gps_accuracy_meters=sim_data_point.accuracy,
                    road_type=sim_data_point.road_type.value,
                    weather_condition=sim_data_point.weather.value,
                    gyroscope_x=sim_data_point.gyroscope_x,
                    gyroscope_y=sim_data_point.gyroscope_y,
                    gyroscope_z=sim_data_point.gyroscope_z,
                    magnetometer_x=sim_data_point.magnetometer_x,
                    magnetometer_y=sim_data_point.magnetometer_y,
                    magnetometer_z=sim_data_point.magnetometer_z,
                    engine_rpm=sim_data_point.engine_rpm,
                    fuel_level=sim_data_point.fuel_level,
                    odometer_reading_km=sim_data_point.odometer_reading
                )
                
                # Save to database
                telematics_repo.create(session, **db_data_point.dict())
                data_points.append(db_data_point)
                
                # Process events
                trip_events = await data_processor.process_data_point(trip_id, sim_data_point)
                for event in trip_events:
                    db_event = DrivingEvent(
                        trip_id=created_trip.id,
                        event_type=event.event_type,
                        timestamp=event.timestamp,
                        severity=event.severity,
                        latitude=event.location.latitude,
                        longitude=event.location.longitude,
                        details=str(event.details)
                    )
                    event_repo.create(session, **db_event.dict())
                    events.append(db_event)
                
                data_count += 1
                if data_count >= 50:  # Limit data points per trip
                    break
            
            await data_collector.stop_collection(trip_id)
            
            # Calculate trip metrics
            trip_metrics = data_processor.calculate_trip_metrics(trip_id)
            if trip_metrics:
                all_trip_metrics.append(trip_metrics)
                
                # Update trip with final metrics
                trip_repo.update(
                    session, created_trip,
                    end_time=trip_metrics.end_time,
                    end_latitude=40.7128 + random.uniform(-0.1, 0.1),
                    end_longitude=-74.0060 + random.uniform(-0.1, 0.1),
                    total_distance_km=trip_metrics.total_distance.value,
                    total_duration_minutes=trip_metrics.total_duration.total_seconds() / 60,
                    average_speed_kmh=trip_metrics.average_speed.value,
                    max_speed_kmh=trip_metrics.max_speed.value,
                    overall_score=trip_metrics.overall_score
                )
                
                # Extract features for ML
                feature_row = feature_extractor.extract_trip_features(
                    created_trip, data_points, events
                )
                feature_repo.create(session, **feature_row.dict())
        
        # Calculate overall risk score
        if all_trip_metrics:
            risk_score = risk_scorer.calculate_risk_score(str(driver.id), all_trip_metrics)
            
            # Save risk score to database
            db_risk_score = RiskScore(
                driver_id=driver.id,
                score=risk_score.score,
                confidence=risk_score.confidence,
                factors=str(risk_score.factors)
            )
            risk_repo.create(session, **db_risk_score.dict())
            
            print(f"  ✅ Generated {num_trips} trips with {len(data_points)} data points")
            print(f"  📊 Risk Score: {risk_score.score:.1f}/100 ({risk_score.get_risk_category()})")
            
            return {
                "driver": driver,
                "vehicle": vehicle,
                "trips": num_trips,
                "risk_score": risk_score,
                "total_data_points": len(data_points),
                "total_events": len(events)
            }
        
        return None


async def main():
    """Initialize database with tables and sample data."""
    print("🚀 Initializing Telematics Insurance Database")
    print("=" * 50)
    
    # Initialize database tables
    init_db()
    
    # Create sample data
    drivers, vehicles, policies = create_sample_data()
    
    # Generate simulation data for each driver
    print("\n🎮 Generating simulation data for all drivers...")
    for driver in drivers:
        await create_simulation_data_for_driver(driver.license_number, num_trips=3)
    
    print("\n🎉 Database initialization complete!")
    print("You can now run the application with:")
    print("  python main.py api")
    print("  python main.py simulation")
    print("\nAvailable drivers:")
    for driver in drivers:
        print(f"  📄 License: {driver.license_number} - {driver.name}")


if __name__ == "__main__":
    main()
