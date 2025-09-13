"""
Database setup script for the Telematics Insurance System.

This script sets up the database with tables and sample data.
Run this script once to initialize your database.
"""

import sys
import os
import asyncio
import random
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.db.engine import init_db, get_session_context
from src.db.repositories import (
    DriverRepository, VehicleRepository, PolicyRepository, 
    LocationRiskProfileRepository, VehicleRiskProfileRepository,
    TripRepository, TelematicsDataPointRepository, DrivingEventRepository,
    RiskScoreRepository, FeatureRowRepository
)
from src.db.models import (
    Driver, Vehicle, Policy, LocationRiskProfile, VehicleRiskProfile,
    Trip, TelematicsDataPoint, DrivingEvent, RiskScore, FeatureRow
)
from src.data_collection.telematics_simulator import (
    TelematicsDataCollector, TelematicsSimulator, TelematicsDataPoint as SimDataPoint,
    RoadType, WeatherCondition, VehicleProfile, LocationRiskProfile as SimLocationProfile
)
from src.data_processing.data_processor import TelematicsDataProcessor
from src.data_processing.features import FeatureExtractor
from src.domain.value_objects import Coordinate


def is_database_empty():
    """Check if the database is empty (no drivers exist)."""
    with get_session_context() as session:
        driver_repo = DriverRepository()
        drivers = driver_repo.get_all(session)
        return len(drivers) == 0


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
        
        # Create sample drivers (15 people with diverse profiles)
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
            ),
            Driver(
                name="Emily Rodriguez",
                email="emily.rodriguez@example.com",
                phone="+1-555-0104",
                license_number="DL789123456",
                date_of_birth=datetime(1992, 3, 10)
            ),
            Driver(
                name="David Wilson",
                email="david.wilson@example.com",
                phone="+1-555-0105",
                license_number="DL321654987",
                date_of_birth=datetime(1987, 7, 18)
            ),
            Driver(
                name="Lisa Anderson",
                email="lisa.anderson@example.com",
                phone="+1-555-0106",
                license_number="DL654987321",
                date_of_birth=datetime(1995, 11, 25)
            ),
            Driver(
                name="Robert Taylor",
                email="robert.taylor@example.com",
                phone="+1-555-0107",
                license_number="DL147258369",
                date_of_birth=datetime(1983, 1, 8)
            ),
            Driver(
                name="Jennifer Brown",
                email="jennifer.brown@example.com",
                phone="+1-555-0108",
                license_number="DL369258147",
                date_of_birth=datetime(1991, 9, 14)
            ),
            Driver(
                name="Michael Davis",
                email="michael.davis@example.com",
                phone="+1-555-0109",
                license_number="DL258147369",
                date_of_birth=datetime(1989, 4, 30)
            ),
            Driver(
                name="Amanda Garcia",
                email="amanda.garcia@example.com",
                phone="+1-555-0110",
                license_number="DL741852963",
                date_of_birth=datetime(1993, 6, 12)
            ),
            Driver(
                name="Christopher Martinez",
                email="christopher.martinez@example.com",
                phone="+1-555-0111",
                license_number="DL963852741",
                date_of_birth=datetime(1986, 10, 7)
            ),
            Driver(
                name="Jessica Thompson",
                email="jessica.thompson@example.com",
                phone="+1-555-0112",
                license_number="DL852741963",
                date_of_birth=datetime(1994, 2, 20)
            ),
            Driver(
                name="Daniel White",
                email="daniel.white@example.com",
                phone="+1-555-0113",
                license_number="DL159753486",
                date_of_birth=datetime(1984, 12, 5)
            ),
            Driver(
                name="Ashley Lee",
                email="ashley.lee@example.com",
                phone="+1-555-0114",
                license_number="DL486159753",
                date_of_birth=datetime(1996, 8, 16)
            ),
            Driver(
                name="Matthew Clark",
                email="matthew.clark@example.com",
                phone="+1-555-0115",
                license_number="DL753486159",
                date_of_birth=datetime(1982, 5, 28)
            )
        ]
        
        created_drivers = []
        for driver in drivers:
            created_driver = driver_repo.create(session, **driver.dict())
            created_drivers.append(created_driver)
        
        # Create sample vehicles (15 diverse vehicles for each driver)
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
            ),
            Vehicle(
                driver_id=created_drivers[3].id,
                make="Ford",
                model="F-150",
                year=2021,
                vin="4HGBH41JXMN109189",
                license_plate="TRK456",
                engine_type="gas",
                safety_rating=4.3,
                theft_probability=0.05
            ),
            Vehicle(
                driver_id=created_drivers[4].id,
                make="BMW",
                model="3 Series",
                year=2020,
                vin="5HGBH41JXMN109190",
                license_plate="BMW789",
                engine_type="gas",
                safety_rating=4.6,
                theft_probability=0.06
            ),
            Vehicle(
                driver_id=created_drivers[5].id,
                make="Nissan",
                model="Altima",
                year=2019,
                vin="6HGBH41JXMN109191",
                license_plate="NIS123",
                engine_type="gas",
                safety_rating=4.2,
                theft_probability=0.03
            ),
            Vehicle(
                driver_id=created_drivers[6].id,
                make="Chevrolet",
                model="Silverado",
                year=2022,
                vin="7HGBH41JXMN109192",
                license_plate="CHE456",
                engine_type="gas",
                safety_rating=4.4,
                theft_probability=0.04
            ),
            Vehicle(
                driver_id=created_drivers[7].id,
                make="Audi",
                model="A4",
                year=2021,
                vin="8HGBH41JXMN109193",
                license_plate="AUD789",
                engine_type="gas",
                safety_rating=4.7,
                theft_probability=0.05
            ),
            Vehicle(
                driver_id=created_drivers[8].id,
                make="Hyundai",
                model="Elantra",
                year=2020,
                vin="9HGBH41JXMN109194",
                license_plate="HYU123",
                engine_type="gas",
                safety_rating=4.3,
                theft_probability=0.02
            ),
            Vehicle(
                driver_id=created_drivers[9].id,
                make="Mercedes-Benz",
                model="C-Class",
                year=2022,
                vin="AHGBH41JXMN109195",
                license_plate="MER456",
                engine_type="gas",
                safety_rating=4.8,
                theft_probability=0.07
            ),
            Vehicle(
                driver_id=created_drivers[10].id,
                make="Subaru",
                model="Outback",
                year=2021,
                vin="BHGBH41JXMN109196",
                license_plate="SUB789",
                engine_type="gas",
                safety_rating=4.5,
                theft_probability=0.02
            ),
            Vehicle(
                driver_id=created_drivers[11].id,
                make="Volkswagen",
                model="Jetta",
                year=2019,
                vin="CHGBH41JXMN109197",
                license_plate="VW123",
                engine_type="gas",
                safety_rating=4.4,
                theft_probability=0.03
            ),
            Vehicle(
                driver_id=created_drivers[12].id,
                make="Lexus",
                model="ES",
                year=2020,
                vin="DHGBH41JXMN109198",
                license_plate="LEX456",
                engine_type="hybrid",
                safety_rating=4.7,
                theft_probability=0.04
            ),
            Vehicle(
                driver_id=created_drivers[13].id,
                make="Mazda",
                model="CX-5",
                year=2021,
                vin="EHGBH41JXMN109199",
                license_plate="MAZ789",
                engine_type="gas",
                safety_rating=4.5,
                theft_probability=0.02
            ),
            Vehicle(
                driver_id=created_drivers[14].id,
                make="Infiniti",
                model="Q50",
                year=2022,
                vin="FHGBH41JXMN109200",
                license_plate="INF123",
                engine_type="gas",
                safety_rating=4.6,
                theft_probability=0.05
            )
        ]
        
        created_vehicles = []
        for vehicle in vehicles:
            created_vehicle = vehicle_repo.create(session, **vehicle.dict())
            created_vehicles.append(created_vehicle)
        
        # Create sample policies (15 policies for all drivers)
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
            ),
            Policy(
                driver_id=created_drivers[3].id,
                vehicle_id=created_vehicles[3].id,
                policy_number="POL-004-2024",
                start_date=datetime.utcnow() - timedelta(days=25),
                end_date=datetime.utcnow() + timedelta(days=340),
                base_premium=1800.0,
                current_premium=1800.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[4].id,
                vehicle_id=created_vehicles[4].id,
                policy_number="POL-005-2024",
                start_date=datetime.utcnow() - timedelta(days=20),
                end_date=datetime.utcnow() + timedelta(days=345),
                base_premium=2000.0,
                current_premium=2000.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[5].id,
                vehicle_id=created_vehicles[5].id,
                policy_number="POL-006-2024",
                start_date=datetime.utcnow() - timedelta(days=10),
                end_date=datetime.utcnow() + timedelta(days=355),
                base_premium=1100.0,
                current_premium=1100.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[6].id,
                vehicle_id=created_vehicles[6].id,
                policy_number="POL-007-2024",
                start_date=datetime.utcnow() - timedelta(days=5),
                end_date=datetime.utcnow() + timedelta(days=360),
                base_premium=1900.0,
                current_premium=1900.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[7].id,
                vehicle_id=created_vehicles[7].id,
                policy_number="POL-008-2024",
                start_date=datetime.utcnow() - timedelta(days=12),
                end_date=datetime.utcnow() + timedelta(days=353),
                base_premium=2100.0,
                current_premium=2100.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[8].id,
                vehicle_id=created_vehicles[8].id,
                policy_number="POL-009-2024",
                start_date=datetime.utcnow() - timedelta(days=18),
                end_date=datetime.utcnow() + timedelta(days=347),
                base_premium=1050.0,
                current_premium=1050.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[9].id,
                vehicle_id=created_vehicles[9].id,
                policy_number="POL-010-2024",
                start_date=datetime.utcnow() - timedelta(days=3),
                end_date=datetime.utcnow() + timedelta(days=362),
                base_premium=2200.0,
                current_premium=2200.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[10].id,
                vehicle_id=created_vehicles[10].id,
                policy_number="POL-011-2024",
                start_date=datetime.utcnow() - timedelta(days=8),
                end_date=datetime.utcnow() + timedelta(days=357),
                base_premium=1300.0,
                current_premium=1300.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[11].id,
                vehicle_id=created_vehicles[11].id,
                policy_number="POL-012-2024",
                start_date=datetime.utcnow() - timedelta(days=14),
                end_date=datetime.utcnow() + timedelta(days=351),
                base_premium=1150.0,
                current_premium=1150.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[12].id,
                vehicle_id=created_vehicles[12].id,
                policy_number="POL-013-2024",
                start_date=datetime.utcnow() - timedelta(days=22),
                end_date=datetime.utcnow() + timedelta(days=343),
                base_premium=1950.0,
                current_premium=1950.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[13].id,
                vehicle_id=created_vehicles[13].id,
                policy_number="POL-014-2024",
                start_date=datetime.utcnow() - timedelta(days=6),
                end_date=datetime.utcnow() + timedelta(days=359),
                base_premium=1250.0,
                current_premium=1250.0,
                status="active"
            ),
            Policy(
                driver_id=created_drivers[14].id,
                vehicle_id=created_vehicles[14].id,
                policy_number="POL-015-2024",
                start_date=datetime.utcnow() - timedelta(days=9),
                end_date=datetime.utcnow() + timedelta(days=356),
                base_premium=2050.0,
                current_premium=2050.0,
                status="active"
            )
        ]
        
        created_policies = []
        for policy in policies:
            created_policy = policy_repo.create(session, **policy.dict())
            created_policies.append(created_policy)
        
        print(f"✅ Created {len(drivers)} drivers, {len(vehicles)} vehicles, {len(policies)} policies")
        print(f"✅ Created {len(locations)} location profiles, {len(vehicle_profiles)} vehicle profiles")
        
        return created_drivers, created_vehicles, created_policies


def create_personalized_simulation_data(license_number: str, driver_index: int):
    """Create personalized telematics data for a specific driver."""
    import random
    from datetime import datetime, timedelta
    from src.data_collection.telematics_simulator import (
        TelematicsDataCollector, TelematicsSimulator, TelematicsDataPoint as SimDataPoint,
        RoadType, WeatherCondition, VehicleProfile, LocationRiskProfile as SimLocationProfile
    )
    from src.data_processing.data_processor import TelematicsDataProcessor
    from src.domain.value_objects import Coordinate
    from src.db.engine import get_session_context
    from src.db.repositories import DriverRepository, TripRepository
    
    # Define different risk profiles for variety
    risk_profiles = [
        # Low risk drivers (indices 0-4)
        {"hard_braking_freq": 0.1, "speeding_freq": 0.05, "night_driving": 0.1, "avg_speed_factor": 0.9},
        {"hard_braking_freq": 0.15, "speeding_freq": 0.08, "night_driving": 0.12, "avg_speed_factor": 0.92},
        {"hard_braking_freq": 0.12, "speeding_freq": 0.06, "night_driving": 0.08, "avg_speed_factor": 0.88},
        {"hard_braking_freq": 0.18, "speeding_freq": 0.1, "night_driving": 0.15, "avg_speed_factor": 0.95},
        {"hard_braking_freq": 0.08, "speeding_freq": 0.04, "night_driving": 0.06, "avg_speed_factor": 0.85},
        
        # Medium risk drivers (indices 5-9)
        {"hard_braking_freq": 0.3, "speeding_freq": 0.2, "night_driving": 0.25, "avg_speed_factor": 1.1},
        {"hard_braking_freq": 0.35, "speeding_freq": 0.22, "night_driving": 0.3, "avg_speed_factor": 1.15},
        {"hard_braking_freq": 0.25, "speeding_freq": 0.18, "night_driving": 0.2, "avg_speed_factor": 1.05},
        {"hard_braking_freq": 0.4, "speeding_freq": 0.25, "night_driving": 0.35, "avg_speed_factor": 1.2},
        {"hard_braking_freq": 0.28, "speeding_freq": 0.15, "night_driving": 0.22, "avg_speed_factor": 1.08},
        
        # High risk drivers (indices 10-14)
        {"hard_braking_freq": 0.6, "speeding_freq": 0.45, "night_driving": 0.4, "avg_speed_factor": 1.3},
        {"hard_braking_freq": 0.7, "speeding_freq": 0.5, "night_driving": 0.45, "avg_speed_factor": 1.35},
        {"hard_braking_freq": 0.55, "speeding_freq": 0.4, "night_driving": 0.35, "avg_speed_factor": 1.25},
        {"hard_braking_freq": 0.8, "speeding_freq": 0.6, "night_driving": 0.5, "avg_speed_factor": 1.4},
        {"hard_braking_freq": 0.65, "speeding_freq": 0.48, "night_driving": 0.42, "avg_speed_factor": 1.32}
    ]
    
    profile = risk_profiles[driver_index % len(risk_profiles)]
    
    with get_session_context() as session:
        driver_repo = DriverRepository()
        trip_repo = TripRepository()
        
        # Get driver by license number
        driver = driver_repo.get_by_license(session, license_number)
        if not driver:
            print(f"   ⚠️  Driver with license {license_number} not found")
            return
        
        # Get driver's vehicle for trip creation
        from src.db.repositories import VehicleRepository
        vehicle_repo = VehicleRepository()
        vehicles = vehicle_repo.get_by_driver(session, driver.id)
        
        if not vehicles:
            print(f"   ⚠️  No vehicle found for driver {license_number}")
            return
            
        vehicle = vehicles[0]  # Use first vehicle
        
        # Create multiple trips with varying characteristics
        num_trips = random.randint(8, 25)  # Vary number of trips per driver
        
        for trip_num in range(num_trips):
            # Generate trip with personalized characteristics
            trip_data = generate_personalized_trip_data(profile, trip_num, driver_index)
            trip_data["driver_id"] = driver.id
            trip_data["vehicle_id"] = vehicle.id
            
            # Create trip record (remove event count fields)
            trip_data_clean = {k: v for k, v in trip_data.items() if k not in [
                'hard_braking_events', 'hard_acceleration_events', 'speeding_events', 'sharp_turn_events'
            ]}
            trip = trip_repo.create(session, **trip_data_clean)
            
            # Create individual driving events
            from src.db.repositories import DrivingEventRepository
            event_repo = DrivingEventRepository()
            
            events_to_create = [
                ('hard_brake', trip_data['hard_braking_events']),
                ('hard_acceleration', trip_data['hard_acceleration_events']),
                ('speeding', trip_data['speeding_events']),
                ('sharp_turn', trip_data['sharp_turn_events'])
            ]
            
            for event_type, count in events_to_create:
                for _ in range(count):
                    # Create random timestamp within trip duration
                    trip_duration = (trip_data['end_time'] - trip_data['start_time']).total_seconds() / 60
                    random_offset = random.uniform(0, trip_duration)
                    event_time = trip_data['start_time'] + timedelta(minutes=random_offset)
                    
                    event_repo.create(session, **{
                        'trip_id': trip.id,
                        'event_type': event_type,
                        'timestamp': event_time,
                        'severity': random.uniform(0.3, 1.0),
                        'latitude': trip_data['start_latitude'] + random.uniform(-0.001, 0.001),
                        'longitude': trip_data['start_longitude'] + random.uniform(-0.001, 0.001),
                        'details': f'{{"speed": {random.uniform(20, 80)}, "acceleration": {random.uniform(-5, 5)}}}'
                    })


def generate_personalized_trip_data(risk_profile: dict, trip_num: int, driver_index: int) -> dict:
    """Generate personalized trip data based on risk profile."""
    import random
    import numpy as np
    from datetime import datetime, timedelta
    
    # Base trip characteristics
    base_distance = random.uniform(5, 50)  # km
    base_duration = random.uniform(15, 120)  # minutes
    
    # Apply risk profile adjustments
    avg_speed = (base_distance / (base_duration / 60)) * risk_profile["avg_speed_factor"]
    max_speed = avg_speed * random.uniform(1.2, 1.8)
    
    # Calculate events based on risk profile
    hard_braking_events = max(0, int(np.random.poisson(risk_profile["hard_braking_freq"] * base_distance)))
    hard_acceleration_events = max(0, int(np.random.poisson(risk_profile["hard_braking_freq"] * 0.7 * base_distance)))
    speeding_events = max(0, int(np.random.poisson(risk_profile["speeding_freq"] * base_distance)))
    sharp_turn_events = max(0, int(np.random.poisson(risk_profile["hard_braking_freq"] * 0.5 * base_distance)))
    
    # Calculate overall score (lower score for higher risk)
    base_score = 100
    base_score -= hard_braking_events * 3
    base_score -= hard_acceleration_events * 2
    base_score -= speeding_events * 4
    base_score -= sharp_turn_events * 2
    overall_score = max(20, min(100, base_score + random.uniform(-5, 5)))
    
    # Random start location (vary by driver)
    lat_base = 40.7128 + (driver_index * 0.1)  # Spread drivers across different areas
    lon_base = -74.0060 + (driver_index * 0.1)
    
    start_lat = lat_base + random.uniform(-0.05, 0.05)
    start_lon = lon_base + random.uniform(-0.05, 0.05)
    end_lat = start_lat + random.uniform(-0.02, 0.02)
    end_lon = start_lon + random.uniform(-0.02, 0.02)
    
    # Trip timing (some drivers drive more at night based on risk profile)
    days_ago = random.randint(1, 30)
    base_hour = 9 if random.random() > risk_profile["night_driving"] else 22
    start_time = datetime.utcnow() - timedelta(days=days_ago, hours=random.randint(-2, 2)) + timedelta(hours=base_hour)
    end_time = start_time + timedelta(minutes=base_duration)
    
    return {
        "driver_id": None,  # Will be set by the caller
        "vehicle_id": None,  # Will be set by the caller
        "start_time": start_time,
        "end_time": end_time,
        "start_latitude": start_lat,
        "start_longitude": start_lon,
        "end_latitude": end_lat,
        "end_longitude": end_lon,
        "total_distance_km": base_distance,
        "total_duration_minutes": base_duration,
        "average_speed_kmh": avg_speed,
        "max_speed_kmh": max_speed,
        "hard_braking_events": hard_braking_events,
        "hard_acceleration_events": hard_acceleration_events,
        "speeding_events": speeding_events,
        "sharp_turn_events": sharp_turn_events,
        "overall_score": overall_score
    }


def calculate_and_store_risk_scores():
    """Calculate and store risk scores for all drivers."""
    from src.db.engine import get_session_context
    from src.db.repositories import DriverRepository, TripRepository, RiskScoreRepository
    from src.risk_scoring.risk_scorer import RiskScoringService
    from src.data_processing.data_processor import TelematicsDataProcessor
    import json
    
    risk_scorer = RiskScoringService()
    data_processor = TelematicsDataProcessor()
    
    with get_session_context() as session:
        driver_repo = DriverRepository()
        trip_repo = TripRepository()
        risk_repo = RiskScoreRepository()
        
        drivers = driver_repo.get_all(session)
        
        for driver in drivers:
            trips = trip_repo.get_by_driver(session, driver.id)
            
            if trips:
                # Convert trips to trip metrics for risk scoring
                trip_metrics = []
                from src.db.repositories import DrivingEventRepository
                event_repo = DrivingEventRepository()
                
                for trip in trips:
                    # Get driving events for this trip
                    events = event_repo.get_by_trip(session, trip.id)
                    
                    # Count events by type
                    event_counts = {'hard_brake': 0, 'hard_acceleration': 0, 'speeding': 0, 'sharp_turn': 0}
                    for event in events:
                        if event.event_type in event_counts:
                            event_counts[event.event_type] += 1
                    
                    # Create a simple trip metric from trip data
                    from src.data_processing.data_processor import TripMetrics
                    
                    from src.domain.value_objects import Distance, Speed
                    from datetime import timedelta
                    
                    metric = TripMetrics(
                        trip_id=str(trip.id),
                        driver_id=str(trip.driver_id),
                        start_time=trip.start_time,
                        end_time=trip.end_time,
                        total_distance=Distance(trip.total_distance_km, 'km'),
                        total_duration=timedelta(minutes=trip.total_duration_minutes),
                        average_speed=Speed(trip.average_speed_kmh, 'km/h'),
                        max_speed=Speed(trip.max_speed_kmh, 'km/h'),
                        hard_braking_events=event_counts['hard_brake'],
                        hard_acceleration_events=event_counts['hard_acceleration'],
                        speeding_events=event_counts['speeding'],
                        sharp_turn_events=event_counts['sharp_turn'],
                        night_driving_percentage=10.0,  # Mock data
                        rush_hour_percentage=20.0,      # Mock data
                        overall_score=trip.overall_score,
                        highway_driving_percentage=30.0,     # Mock data
                        city_driving_percentage=50.0,        # Mock data
                        residential_driving_percentage=20.0, # Mock data
                        weather_impact_score=85.0,  # Mock data
                        road_quality_score=80.0,    # Mock data
                        traffic_density_score=75.0  # Mock data
                    )
                    trip_metrics.append(metric)
                
                # Calculate risk score
                risk_score = risk_scorer.calculate_risk_score(str(driver.id), trip_metrics)
                
                # Store in database
                risk_repo.create(session, **{
                    "driver_id": driver.id,
                    "score": risk_score.score,
                    "confidence": risk_score.confidence,
                    "factors": json.dumps(risk_score.factors) if isinstance(risk_score.factors, list) else risk_score.factors,
                    "calculated_at": risk_score.calculated_at
                })


def calculate_and_update_premiums():
    """Calculate and update premiums for all drivers."""
    from src.db.engine import get_session_context
    from src.db.repositories import DriverRepository, PolicyRepository, RiskScoreRepository
    from src.pricing_engine.pricing_calculator import PricingService, PricingFactors
    
    pricing_service = PricingService()
    
    with get_session_context() as session:
        driver_repo = DriverRepository()
        policy_repo = PolicyRepository()
        risk_repo = RiskScoreRepository()
        
        drivers = driver_repo.get_all(session)
        
        for driver in drivers:
            policies = policy_repo.get_by_driver(session, driver.id)
            risk_scores = risk_repo.get_by_driver(session, driver.id)
            
            if policies and risk_scores:
                policy = policies[0]  # Use first policy
                latest_risk_score = max(risk_scores, key=lambda rs: rs.calculated_at)
                
                # Create pricing factors
                pricing_factors = PricingFactors(
                    base_premium=policy.base_premium,
                    risk_score=latest_risk_score.score,  # Already on 0-100 scale
                    driving_experience_years=5,
                    vehicle_age=3,
                    annual_mileage=15000,
                    location_risk_factor=1.0,
                    time_since_last_claim=24,
                    credit_score=750,
                    policy_duration_months=12,
                    vehicle_safety_rating=4.0,
                    claims_history=0,
                    age=30,
                    marital_status="single",
                    education_level="college",
                    occupation="professional"
                )
                
                # Calculate new premium
                pricing_result = pricing_service.calculate_premium(pricing_factors)
                
                # Update policy with new premium
                policy.current_premium = pricing_result.adjusted_premium
                session.add(policy)
                session.commit()


async def main():
    """Initialize database with tables and sample data."""
    print("🚀 Initializing Telematics Insurance Database")
    print("=" * 50)
    
    # Initialize database tables first
    init_db()
    
    # Check if database is already populated
    if not is_database_empty():
        print("⚠️  Database already contains data. Skipping initialization to prevent duplicates.")
        print("💡 To reset the database, delete the file: data/telematics_insurance.db")
        return
    
    # Create sample data
    drivers, vehicles, policies = create_sample_data()
    
    # Create personalized simulation data for all drivers
    print("\n🎯 Creating personalized simulation data for drivers...")
    driver_license_numbers = [
        "DL123456789", "DL987654321", "DL456789123", "DL789123456", "DL321654987",
        "DL654987321", "DL147258369", "DL369258147", "DL258147369", "DL741852963",
        "DL963852741", "DL852741963", "DL159753486", "DL486159753", "DL753486159"
    ]
    
    for i, license_number in enumerate(driver_license_numbers):
        create_personalized_simulation_data(license_number, i)
        print(f"   ✅ Created personalized data for driver {i+1}/{len(driver_license_numbers)}")
    
    # Calculate and store risk scores for all drivers
    print("\n🧠 Calculating risk scores for all drivers...")
    calculate_and_store_risk_scores()
    print("   ✅ Risk scores calculated and stored")
    
    # Calculate and update premiums
    print("\n💰 Calculating personalized premiums...")
    calculate_and_update_premiums()
    print("   ✅ Premiums calculated and updated")
    
    print("\n🎉 Database initialization complete!")
    print("You can now run the application with:")
    print("  python main.py api")
    print("  python main.py simulation")
    print(f"\n📊 Created {len(drivers)} drivers with complete data")
    print("Available drivers:")
    driver_license_numbers = [
        "DL123456789", "DL987654321", "DL456789123", "DL789123456", "DL321654987",
        "DL654987321", "DL147258369", "DL369258147", "DL258147369", "DL741852963",
        "DL963852741", "DL852741963", "DL159753486", "DL486159753", "DL753486159"
    ]
    for i, license_number in enumerate(driver_license_numbers, 1):
        print(f"  {i:2d}. License: {license_number}")


if __name__ == "__main__":
    asyncio.run(main())
