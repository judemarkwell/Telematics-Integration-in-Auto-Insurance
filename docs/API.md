# API Documentation

## Overview

The Telematics Insurance System provides a comprehensive REST API for managing drivers, vehicles, policies, and telematics data. The API follows RESTful principles and provides automatic OpenAPI documentation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for demonstration purposes. In production, JWT-based authentication would be implemented.

## API Endpoints

### Driver Management

#### Create Driver
```http
POST /drivers
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "+1234567890",
  "license_number": "DL123456789",
  "date_of_birth": "1990-01-01T00:00:00Z"
}
```

**Response:**
```json
{
  "driver_id": "uuid-string",
  "name": "John Doe",
  "email": "john.doe@example.com",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### Get Driver
```http
GET /drivers/{driver_id}
```

**Response:**
```json
{
  "driver_id": "uuid-string",
  "name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "+1234567890",
  "license_number": "DL123456789",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Vehicle Management

#### Register Vehicle
```http
POST /vehicles
Content-Type: application/json

{
  "driver_id": "uuid-string",
  "make": "Toyota",
  "model": "Camry",
  "year": 2020,
  "vin": "1HGBH41JXMN109186",
  "license_plate": "ABC123"
}
```

**Response:**
```json
{
  "vehicle_id": "uuid-string",
  "make": "Toyota",
  "model": "Camry",
  "year": 2020
}
```

### Policy Management

#### Create Policy
```http
POST /policies
Content-Type: application/json

{
  "driver_id": "uuid-string",
  "vehicle_id": "uuid-string",
  "base_premium": 1000.0
}
```

**Response:**
```json
{
  "policy_id": "uuid-string",
  "policy_number": "POL-123456",
  "base_premium": 1000.0,
  "current_premium": 1000.0,
  "status": "active"
}
```

### Trip Management

#### Start Trip
```http
POST /trips/start
Content-Type: application/json

{
  "driver_id": "uuid-string",
  "vehicle_id": "uuid-string",
  "start_latitude": 40.7128,
  "start_longitude": -74.0060
}
```

**Response:**
```json
{
  "trip_id": "uuid-string",
  "driver_id": "uuid-string",
  "vehicle_id": "uuid-string",
  "started_at": "2024-01-01T00:00:00Z"
}
```

#### Stop Trip
```http
POST /trips/{trip_id}/stop
```

**Response:**
```json
{
  "trip_id": "uuid-string",
  "duration_minutes": 45,
  "distance_km": 25.5,
  "average_speed": 34.0,
  "max_speed": 65.0,
  "overall_score": 85.5,
  "events": {
    "hard_braking": 2,
    "hard_acceleration": 1,
    "speeding": 0,
    "sharp_turns": 1
  },
  "stopped_at": "2024-01-01T00:45:00Z"
}
```

### Dashboard & Analytics

#### Get Dashboard Summary
```http
GET /dashboard/{driver_id}
```

**Response:**
```json
{
  "driver_name": "John Doe",
  "current_risk_score": 65.5,
  "risk_category": "medium",
  "current_premium": 950.0,
  "premium_change": -50.0,
  "total_trips": 25,
  "total_distance": 1250.5,
  "average_score": 82.3,
  "last_trip_date": "2024-01-01T00:45:00Z",
  "policy_status": "active"
}
```

#### Get Recent Trips
```http
GET /trips/{driver_id}/recent?limit=10
```

**Response:**
```json
[
  {
    "trip_id": "uuid-string",
    "date": "2024-01-01T00:45:00Z",
    "duration_minutes": 45,
    "distance_km": 25.5,
    "average_speed": 34.0,
    "max_speed": 65.0,
    "score": 85.5,
    "events_count": 4,
    "route_summary": "08:00 - 08:45"
  }
]
```

#### Get Risk Score
```http
GET /risk-score/{driver_id}
```

**Response:**
```json
{
  "driver_id": "uuid-string",
  "score": 65.5,
  "confidence": 0.85,
  "factors": [
    "Frequent hard braking",
    "Good speed limit compliance"
  ],
  "calculated_at": "2024-01-01T00:45:00Z"
}
```

#### Get Driving Insights
```http
GET /driving-insights/{driver_id}
```

**Response:**
```json
{
  "safety_score": 75.0,
  "efficiency_score": 80.0,
  "consistency_score": 70.0,
  "improvement_areas": [
    "Reduce hard braking",
    "Smoother acceleration"
  ],
  "strengths": [
    "Speed limit compliance",
    "Consistent driving"
  ],
  "recommendations": [
    "Increase following distance to allow for gentler stops",
    "Gradually increase speed when starting from stops"
  ]
}
```

#### Get Premium History
```http
GET /premium-history/{driver_id}?months=12
```

**Response:**
```json
[
  {
    "date": "2024-01-01T00:00:00Z",
    "premium": 1000.0,
    "risk_score": 70.0,
    "factors": ["Standard pricing"]
  }
]
```

#### Get Risk Breakdown
```http
GET /risk-breakdown/{driver_id}
```

**Response:**
```json
{
  "overall_score": 65.5,
  "confidence": 0.85,
  "category": "medium",
  "factors": [
    "Frequent hard braking",
    "Good speed limit compliance"
  ],
  "event_rate": 2.5,
  "safety_events": {
    "hard_braking": 15,
    "hard_acceleration": 8,
    "speeding": 3,
    "sharp_turns": 5
  },
  "calculated_at": "2024-01-01T00:45:00Z"
}
```

### System Endpoints

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

#### API Documentation
```http
GET /docs
```

Returns interactive Swagger UI documentation.

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes

- `400` - Bad Request: Invalid input data
- `404` - Not Found: Resource not found
- `422` - Unprocessable Entity: Validation error
- `500` - Internal Server Error: Server error

### Example Error Response
```json
{
  "detail": "Driver not found",
  "error_code": "DRIVER_NOT_FOUND",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, rate limiting would be added to prevent abuse.

## Pagination

For endpoints that return lists, pagination parameters are supported:

- `limit`: Maximum number of items to return (default: 10)
- `offset`: Number of items to skip (default: 0)

Example:
```http
GET /trips/{driver_id}/recent?limit=20&offset=0
```

## Data Models

### Driver
```json
{
  "id": "uuid",
  "name": "string",
  "email": "string",
  "phone": "string",
  "license_number": "string",
  "date_of_birth": "datetime",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Vehicle
```json
{
  "id": "uuid",
  "driver_id": "uuid",
  "make": "string",
  "model": "string",
  "year": "integer",
  "vin": "string",
  "license_plate": "string",
  "created_at": "datetime"
}
```

### Policy
```json
{
  "id": "uuid",
  "driver_id": "uuid",
  "vehicle_id": "uuid",
  "policy_number": "string",
  "start_date": "datetime",
  "end_date": "datetime",
  "base_premium": "float",
  "current_premium": "float",
  "status": "string",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Risk Score
```json
{
  "id": "uuid",
  "driver_id": "uuid",
  "score": "float",
  "confidence": "float",
  "factors": ["string"],
  "calculated_at": "datetime"
}
```

## SDK Examples

### Python
```python
import requests

# Create a driver
response = requests.post("http://localhost:8000/drivers", json={
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1234567890",
    "license_number": "DL123456789"
})
driver = response.json()

# Start a trip
response = requests.post("http://localhost:8000/trips/start", json={
    "driver_id": driver["driver_id"],
    "vehicle_id": "vehicle-uuid"
})
trip = response.json()

# Get dashboard summary
response = requests.get(f"http://localhost:8000/dashboard/{driver['driver_id']}")
dashboard = response.json()
```

### JavaScript
```javascript
// Create a driver
const driverResponse = await fetch('http://localhost:8000/drivers', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '+1234567890',
    license_number: 'DL123456789'
  })
});
const driver = await driverResponse.json();

// Start a trip
const tripResponse = await fetch('http://localhost:8000/trips/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    driver_id: driver.driver_id,
    vehicle_id: 'vehicle-uuid'
  })
});
const trip = await tripResponse.json();
```

## Testing

### Using curl
```bash
# Create a driver
curl -X POST "http://localhost:8000/drivers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1234567890",
    "license_number": "DL123456789"
  }'

# Get dashboard summary
curl "http://localhost:8000/dashboard/{driver_id}"
```

### Using Postman
Import the OpenAPI specification from `/docs` into Postman for easy testing.

## WebSocket Support

Currently, the API only supports HTTP REST endpoints. WebSocket support for real-time updates would be a future enhancement.

