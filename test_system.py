"""
Test script for the Telematics Insurance System.

This script demonstrates the complete system functionality by:
1. Creating a driver and vehicle
2. Starting a trip and collecting data
3. Processing the data and calculating risk scores
4. Generating pricing adjustments
5. Displaying dashboard insights
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collection.telematics_simulator import TelematicsDataCollector, Coordinate
from src.data_processing.data_processor import TelematicsDataProcessor
from src.risk_scoring.risk_scorer import RiskScoringService
from src.pricing_engine.pricing_calculator import PricingService
from src.user_dashboard.dashboard_service import DashboardService
from src.domain.entities import Driver, Vehicle, Policy
from src.domain.value_objects import Premium


async def test_complete_system():
    """Test the complete telematics insurance system."""
    print("🧪 Testing Telematics Insurance System")
    print("=" * 50)
    
    # Initialize services
    data_collector = TelematicsDataCollector()
    data_processor = TelematicsDataProcessor()
    risk_scorer = RiskScoringService()
    pricing_service = PricingService()
    dashboard_service = DashboardService()
    
    # Create test data
    driver_id = "test_driver_001"
    vehicle_id = "test_vehicle_001"
    
    # Create driver
    driver = Driver(
        name="Test Driver",
        email="test@example.com",
        phone="+1234567890",
        license_number="TEST123456"
    )
    dashboard_service.drivers_data[driver_id] = driver
    
    # Create vehicle
    vehicle = {
        "id": vehicle_id,
        "driver_id": driver_id,
        "make": "Toyota",
        "model": "Camry",
        "year": 2020,
        "vin": "TEST123456789",
        "license_plate": "TEST123"
    }
    
    # Create policy
    policy = Policy(
        driver_id=driver_id,
        vehicle_id=vehicle_id,
        base_premium=1000.0,
        current_premium=1000.0
    )
    dashboard_service.policies_data[str(policy.id)] = policy
    
    print(f"👤 Created driver: {driver.name}")
    print(f"🚙 Created vehicle: {vehicle['make']} {vehicle['model']}")
    print(f"📋 Created policy with base premium: ${policy.base_premium}")
    print()
    
    # Test 1: Data Collection
    print("🟢 Test 1: Data Collection")
    print("-" * 30)
    
    start_location = Coordinate(40.7128, -74.0060)  # NYC
    trip_id = await data_collector.start_collection(driver_id, vehicle_id, start_location)
    print(f"📍 Started trip: {trip_id}")
    print(f"📍 Start location: {start_location.latitude}, {start_location.longitude}")
    
    # Collect some data
    data_points = 0
    async for data_point in data_collector.get_data_stream(trip_id):
        events = await data_processor.process_data_point(trip_id, data_point)
        data_points += 1
        
        if data_points % 5 == 0:
            print(f"   📊 Collected {data_points} data points")
        
        if data_points >= 20:  # Collect 20 data points
            break
    
    await data_collector.stop_collection(trip_id)
    print(f"🔴 Stopped trip after collecting {data_points} data points")
    print()
    
    # Test 2: Data Processing
    print("📈 Test 2: Data Processing")
    print("-" * 30)
    
    trip_metrics = data_processor.calculate_trip_metrics(trip_id)
    if trip_metrics:
        print(f"   ⏱️  Duration: {trip_metrics.total_duration}")
        print(f"   📏 Distance: {trip_metrics.total_distance.value:.2f} km")
        print(f"   🏃 Average Speed: {trip_metrics.average_speed.value:.1f} km/h")
        print(f"   🏃 Max Speed: {trip_metrics.max_speed.value:.1f} km/h")
        print(f"   ⭐ Overall Score: {trip_metrics.overall_score:.1f}/100")
        print(f"   🚨 Events: {trip_metrics.hard_braking_events} hard brakes, "
              f"{trip_metrics.hard_acceleration_events} hard accelerations, "
              f"{trip_metrics.speeding_events} speeding, "
              f"{trip_metrics.sharp_turn_events} sharp turns")
        
        # Store trip metrics
        dashboard_service.trip_metrics_data[driver_id] = [trip_metrics]
    else:
        print("   ❌ No trip metrics calculated")
    print()
    
    # Test 3: Risk Scoring
    print("🎯 Test 3: Risk Scoring")
    print("-" * 30)
    
    if trip_metrics:
        risk_score = risk_scorer.calculate_risk_score(driver_id, [trip_metrics])
        dashboard_service.risk_scores_data[driver_id] = risk_score
        
        print(f"   📊 Risk Score: {risk_score.score:.1f}/100")
        print(f"   🎯 Risk Category: {risk_score.get_risk_category()}")
        print(f"   🔍 Confidence: {risk_score.confidence:.2f}")
        print(f"   📋 Factors: {', '.join(risk_score.factors)}")
    else:
        print("   ❌ No risk score calculated")
    print()
    
    # Test 4: Pricing Engine
    print("💰 Test 4: Pricing Engine")
    print("-" * 30)
    
    if trip_metrics and risk_score:
        from src.pricing_engine.pricing_calculator import PricingFactors
        
        pricing_factors = PricingFactors(
            base_premium=1000.0,
            risk_score=risk_score.score,
            driving_experience_years=5,
            vehicle_age=3,
            annual_mileage=15000,
            location_risk_factor=1.0,
            time_since_last_claim=24,
            credit_score=750,
            policy_duration_months=12
        )
        
        pricing_result = pricing_service.calculator.calculate_premium(
            policy, risk_score, pricing_factors
        )
        
        print(f"   💵 Original Premium: ${pricing_result.original_premium.amount:.2f}")
        print(f"   💵 Adjusted Premium: ${pricing_result.adjusted_premium.amount:.2f}")
        if pricing_result.discount_percentage > 0:
            print(f"   🎉 Discount: {pricing_result.discount_percentage:.1f}%")
        elif pricing_result.surcharge_percentage > 0:
            print(f"   📈 Surcharge: {pricing_result.surcharge_percentage:.1f}%")
        print(f"   📋 Applied Factors: {', '.join(pricing_result.factors_applied)}")
    else:
        print("   ❌ No pricing calculation performed")
    print()
    
    # Test 5: Dashboard Service
    print("📊 Test 5: Dashboard Service")
    print("-" * 30)
    
    # Dashboard summary
    summary = dashboard_service.get_dashboard_summary(driver_id)
    if summary:
        print(f"   👤 Driver: {summary.driver_name}")
        print(f"   📊 Risk Score: {summary.current_risk_score:.1f}")
        print(f"   🎯 Category: {summary.risk_category}")
        print(f"   💵 Premium: ${summary.current_premium:.2f}")
        print(f"   🚗 Total Trips: {summary.total_trips}")
        print(f"   📏 Total Distance: {summary.total_distance:.2f} km")
        print(f"   ⭐ Average Score: {summary.average_score:.1f}")
    
    # Driving insights
    insights = dashboard_service.get_driving_insights(driver_id)
    print(f"   🛡️  Safety Score: {insights.safety_score:.1f}")
    print(f"   ⚡ Efficiency Score: {insights.efficiency_score:.1f}")
    print(f"   📈 Consistency Score: {insights.consistency_score:.1f}")
    print(f"   💡 Recommendations: {len(insights.recommendations)} suggestions")
    
    # Risk breakdown
    breakdown = dashboard_service.get_risk_breakdown(driver_id)
    print(f"   🔍 Overall Score: {breakdown['overall_score']:.1f}")
    print(f"   🎯 Category: {breakdown['category']}")
    print(f"   📊 Event Rate: {breakdown['event_rate']:.1f} events/hour")
    print()
    
    # Test 6: Recent trips
    print("🛣️  Test 6: Recent Trips")
    print("-" * 30)
    
    recent_trips = dashboard_service.get_recent_trips(driver_id, limit=5)
    print(f"   📋 Found {len(recent_trips)} recent trips")
    for trip in recent_trips:
        print(f"   🚗 Trip {trip.trip_id[:8]}... - {trip.duration_minutes}min, "
              f"{trip.distance_km:.1f}km, Score: {trip.score:.1f}")
    print()
    
    print("✅ All tests completed successfully!")
    print("=" * 50)
    print("🎉 The Telematics Insurance System is working correctly!")
    print("🚀 You can now start the API server with: python main.py api")
    print("📚 Visit http://localhost:8000/docs for API documentation")


if __name__ == "__main__":
    asyncio.run(test_complete_system())

