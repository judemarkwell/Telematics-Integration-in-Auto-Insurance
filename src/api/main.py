"""
FastAPI application for telematics insurance system.

This module provides REST API endpoints for the telematics insurance system,
including data collection, risk scoring, pricing, and dashboard functionality.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import uuid

from ..config.settings import config
from ..data_collection.telematics_simulator import TelematicsDataCollector, Coordinate
from ..data_processing.data_processor import TelematicsDataProcessor
from ..risk_scoring.risk_scorer import RiskScoringService
from ..pricing_engine.pricing_calculator import PricingService
from ..user_dashboard.dashboard_service import DashboardService
from ..domain.entities import Driver, Policy, RiskScore
from ..domain.value_objects import Premium


# Pydantic models for API requests/responses
class DriverCreateRequest(BaseModel):
    name: str
    email: str
    phone: str
    license_number: str
    date_of_birth: Optional[datetime] = None


class VehicleCreateRequest(BaseModel):
    driver_id: str
    make: str
    model: str
    year: int
    vin: str
    license_plate: str


class PolicyCreateRequest(BaseModel):
    driver_id: str
    vehicle_id: str
    base_premium: float


class TripStartRequest(BaseModel):
    driver_id: str
    vehicle_id: str
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None


class DashboardSummaryResponse(BaseModel):
    driver_name: str
    current_risk_score: float
    risk_category: str
    current_premium: float
    premium_change: float
    total_trips: int
    total_distance: float
    average_score: float
    last_trip_date: Optional[datetime]
    policy_status: str


class TripSummaryResponse(BaseModel):
    trip_id: str
    date: datetime
    duration_minutes: int
    distance_km: float
    average_speed: float
    max_speed: float
    score: float
    events_count: int
    route_summary: str


class RiskScoreResponse(BaseModel):
    driver_id: str
    score: float
    confidence: float
    factors: List[str]
    calculated_at: datetime


# Initialize FastAPI app
app = FastAPI(
    title="Telematics Insurance API",
    description="API for telematics-based auto insurance system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_collector = TelematicsDataCollector()
data_processor = TelematicsDataProcessor()
risk_scorer = RiskScoringService()
pricing_service = PricingService()
dashboard_service = DashboardService()

# In-memory storage (in production, use proper database)
drivers_db: Dict[str, Driver] = {}
vehicles_db: Dict[str, Any] = {}
policies_db: Dict[str, Policy] = {}
active_trips: Dict[str, str] = {}  # driver_id -> trip_id


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>Telematics Insurance API</title>
        </head>
        <body>
            <h1>Telematics Insurance API</h1>
            <p>Welcome to the Telematics Insurance API!</p>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation.</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><strong>POST /drivers</strong> - Create a new driver</li>
                <li><strong>POST /vehicles</strong> - Register a vehicle</li>
                <li><strong>POST /policies</strong> - Create an insurance policy</li>
                <li><strong>POST /trips/start</strong> - Start a driving trip</li>
                <li><strong>POST /trips/{trip_id}/stop</strong> - Stop a driving trip</li>
                <li><strong>GET /dashboard/{driver_id}</strong> - Get dashboard summary</li>
                <li><strong>GET /risk-score/{driver_id}</strong> - Get current risk score</li>
                <li><strong>GET /trips/{driver_id}/recent</strong> - Get recent trips</li>
            </ul>
        </body>
    </html>
    """


@app.post("/drivers")
async def create_driver(request: DriverCreateRequest):
    """Create a new driver."""
    driver = Driver(
        name=request.name,
        email=request.email,
        phone=request.phone,
        license_number=request.license_number,
        date_of_birth=request.date_of_birth
    )
    
    drivers_db[str(driver.id)] = driver
    dashboard_service.drivers_data[str(driver.id)] = driver
    
    return {
        "driver_id": str(driver.id),
        "name": driver.name,
        "email": driver.email,
        "created_at": driver.created_at
    }


@app.post("/vehicles")
async def create_vehicle(request: VehicleCreateRequest):
    """Register a new vehicle."""
    if request.driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    vehicle = {
        "id": str(uuid.uuid4()),
        "driver_id": request.driver_id,
        "make": request.make,
        "model": request.model,
        "year": request.year,
        "vin": request.vin,
        "license_plate": request.license_plate,
        "created_at": datetime.utcnow()
    }
    
    vehicles_db[vehicle["id"]] = vehicle
    
    return {
        "vehicle_id": vehicle["id"],
        "make": vehicle["make"],
        "model": vehicle["model"],
        "year": vehicle["year"]
    }


@app.post("/policies")
async def create_policy(request: PolicyCreateRequest):
    """Create an insurance policy."""
    if request.driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if request.vehicle_id not in vehicles_db:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    
    policy = Policy(
        driver_id=request.driver_id,
        vehicle_id=request.vehicle_id,
        base_premium=request.base_premium,
        current_premium=request.base_premium
    )
    
    policies_db[str(policy.id)] = policy
    dashboard_service.policies_data[str(policy.id)] = policy
    
    return {
        "policy_id": str(policy.id),
        "policy_number": policy.policy_number,
        "base_premium": policy.base_premium,
        "current_premium": policy.current_premium,
        "status": policy.status
    }


@app.post("/trips/start")
async def start_trip(request: TripStartRequest, background_tasks: BackgroundTasks):
    """Start a new driving trip."""
    if request.driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    if request.driver_id in active_trips:
        raise HTTPException(status_code=400, detail="Driver already has an active trip")
    
    # Create start location if provided
    start_location = None
    if request.start_latitude and request.start_longitude:
        start_location = Coordinate(request.start_latitude, request.start_longitude)
    
    # Start data collection
    trip_id = await data_collector.start_collection(
        request.driver_id, 
        request.vehicle_id,
        start_location
    )
    
    active_trips[request.driver_id] = trip_id
    
    # Start background data processing
    background_tasks.add_task(process_trip_data, trip_id)
    
    return {
        "trip_id": trip_id,
        "driver_id": request.driver_id,
        "vehicle_id": request.vehicle_id,
        "started_at": datetime.utcnow()
    }


@app.post("/trips/{trip_id}/stop")
async def stop_trip(trip_id: str):
    """Stop a driving trip and calculate final metrics."""
    # Find driver for this trip
    driver_id = None
    for driver, active_trip_id in active_trips.items():
        if active_trip_id == trip_id:
            driver_id = driver
            break
    
    if not driver_id:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    # Stop data collection
    await data_collector.stop_collection(trip_id)
    del active_trips[driver_id]
    
    # Calculate final trip metrics
    trip_metrics = data_processor.calculate_trip_metrics(trip_id)
    
    if trip_metrics:
        # Store trip metrics
        if driver_id not in dashboard_service.trip_metrics_data:
            dashboard_service.trip_metrics_data[driver_id] = []
        dashboard_service.trip_metrics_data[driver_id].append(trip_metrics)
        
        # Update risk score
        all_trip_metrics = dashboard_service.trip_metrics_data[driver_id]
        risk_score = risk_scorer.calculate_risk_score(driver_id, all_trip_metrics)
        dashboard_service.risk_scores_data[driver_id] = risk_score
        
        # Clear processed data
        data_processor.clear_trip_data(trip_id)
        
        return {
            "trip_id": trip_id,
            "duration_minutes": int(trip_metrics.total_duration.total_seconds() / 60),
            "distance_km": trip_metrics.total_distance.value,
            "average_speed": trip_metrics.average_speed.value,
            "max_speed": trip_metrics.max_speed.value,
            "overall_score": trip_metrics.overall_score,
            "events": {
                "hard_braking": trip_metrics.hard_braking_events,
                "hard_acceleration": trip_metrics.hard_acceleration_events,
                "speeding": trip_metrics.speeding_events,
                "sharp_turns": trip_metrics.sharp_turn_events
            },
            "stopped_at": datetime.utcnow()
        }
    
    return {"message": "Trip stopped", "trip_id": trip_id}


@app.get("/dashboard/{driver_id}", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(driver_id: str):
    """Get dashboard summary for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    summary = dashboard_service.get_dashboard_summary(driver_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Dashboard data not found")
    
    return summary


@app.get("/trips/{driver_id}/recent", response_model=List[TripSummaryResponse])
async def get_recent_trips(driver_id: str, limit: int = 10):
    """Get recent trips for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    trips = dashboard_service.get_recent_trips(driver_id, limit)
    return trips


@app.get("/risk-score/{driver_id}", response_model=RiskScoreResponse)
async def get_risk_score(driver_id: str):
    """Get current risk score for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    risk_score = dashboard_service.risk_scores_data.get(driver_id)
    if not risk_score:
        raise HTTPException(status_code=404, detail="Risk score not found")
    
    return risk_score


@app.get("/driving-insights/{driver_id}")
async def get_driving_insights(driver_id: str):
    """Get driving behavior insights for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    insights = dashboard_service.get_driving_insights(driver_id)
    return insights


@app.get("/premium-history/{driver_id}")
async def get_premium_history(driver_id: str, months: int = 12):
    """Get premium history for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    history = dashboard_service.get_premium_history(driver_id, months)
    return history


@app.get("/risk-breakdown/{driver_id}")
async def get_risk_breakdown(driver_id: str):
    """Get detailed risk score breakdown for a driver."""
    if driver_id not in drivers_db:
        raise HTTPException(status_code=404, detail="Driver not found")
    
    breakdown = dashboard_service.get_risk_breakdown(driver_id)
    return breakdown


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }


async def process_trip_data(trip_id: str):
    """Background task to process telematics data during a trip."""
    try:
        async for data_point in data_collector.get_data_stream(trip_id):
            events = await data_processor.process_data_point(trip_id, data_point)
            
            # In a real implementation, you might want to:
            # - Store events in database
            # - Send real-time notifications
            # - Update risk scores incrementally
            
            # For now, just log the events
            if events:
                print(f"Trip {trip_id}: Detected {len(events)} events")
    
    except Exception as e:
        print(f"Error processing trip {trip_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )

