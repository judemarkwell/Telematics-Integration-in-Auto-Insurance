"""
Dashboard Service for Telematics Insurance.

This module provides services for generating dashboard data and insights
for users based on their driving behavior and risk scores.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from uuid import UUID

from ..db.engine import get_session_context
from ..db.repositories import (
    DriverRepository, PolicyRepository, RiskScoreRepository, 
    TripRepository, DrivingEventRepository
)


class DashboardService:
    """Service for generating dashboard data and insights."""
    
    def __init__(self):
        """Initialize the dashboard service."""
        pass
    
    def get_dashboard_summary(self, driver_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dashboard summary for a driver from the database.
        
        Args:
            driver_id: Driver identifier (UUID string)
            
        Returns:
            Dictionary with dashboard data or None if driver not found
        """
        with get_session_context() as session:
            driver_repo = DriverRepository()
            policy_repo = PolicyRepository()
            risk_repo = RiskScoreRepository()
            trip_repo = TripRepository()

            # Convert string UUID to UUID object
            try:
                driver_uuid = UUID(driver_id)
            except ValueError:
                return None
                
            driver = driver_repo.get_by_id(session, driver_uuid)
            if not driver:
                return None

            policies = policy_repo.get_by_driver(session, driver.id)
            policy = policies[0] if policies else None

            risk_scores = risk_repo.get_by_driver(session, driver.id)
            latest_risk_score = max(risk_scores, key=lambda rs: rs.calculated_at) if risk_scores else None

            trips = trip_repo.get_by_driver(session, driver.id)

            total_trips = len(trips)
            total_distance = sum(trip.total_distance_km for trip in trips)
            average_score = sum(trip.overall_score for trip in trips) / total_trips if total_trips > 0 else 0.0
            last_trip_date = max(trip.end_time for trip in trips) if trips else None

            current_premium = policy.current_premium if policy else 0.0
            base_premium = policy.base_premium if policy else 0.0
            premium_change = current_premium - base_premium

            # Determine risk category from score
            risk_category_str = "medium"
            if latest_risk_score:
                score = latest_risk_score.score
                if score <= 30:
                    risk_category_str = "low"
                elif score <= 60:
                    risk_category_str = "medium"
                elif score <= 80:
                    risk_category_str = "high"
                else:
                    risk_category_str = "very_high"

            return {
                "driver_name": driver.name,
                "current_risk_score": latest_risk_score.score if latest_risk_score else 75.0,
                "risk_category": risk_category_str,
                "current_premium": current_premium,
                "premium_change": premium_change,
                "total_trips": total_trips,
                "total_distance": round(total_distance, 2),
                "average_score": round(average_score, 1),
                "last_trip_date": last_trip_date.isoformat() if last_trip_date else None,
                "policy_status": policy.status if policy else "inactive"
            }
    
    def get_recent_trips(self, driver_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trips for a driver."""
        with get_session_context() as session:
            trip_repo = TripRepository()
            
            try:
                driver_uuid = UUID(driver_id)
            except ValueError:
                return []
                
            trips = trip_repo.get_by_driver(session, driver_uuid, limit=limit)
            
            recent_trips = []
            for trip in trips:
                recent_trips.append({
                    "trip_id": str(trip.id),
                    "date": trip.start_time.isoformat(),
                    "duration_minutes": round(trip.total_duration_minutes),
                    "distance_km": round(trip.total_distance_km, 1),
                    "average_speed": round(trip.average_speed_kmh, 1),
                    "max_speed": round(trip.max_speed_kmh, 1),
                    "score": round(trip.overall_score, 1),
                    "route_summary": f"Trip on {trip.start_time.strftime('%Y-%m-%d %H:%M')}"
                })
            
            return recent_trips
    
    def get_driving_insights(self, driver_id: str) -> Dict[str, Any]:
        """Get driving insights for a driver."""
        with get_session_context() as session:
            trip_repo = TripRepository()
            event_repo = DrivingEventRepository()
            
            try:
                driver_uuid = UUID(driver_id)
            except ValueError:
                return {"total_trips": 0, "average_score": 0.0, "improvement_trend": "no_data", "recommendations": []}
                
            trips = trip_repo.get_by_driver(session, driver_uuid)
            
            if not trips:
                return {
                    "total_trips": 0,
                    "average_score": 0.0,
                    "improvement_trend": "no_data",
                    "recommendations": ["Start driving to get insights!"]
                }
            
            total_trips = len(trips)
            average_score = sum(trip.overall_score for trip in trips) / total_trips
            
            # Count driving events
            total_events = 0
            for trip in trips:
                events = event_repo.get_by_trip(session, trip.id)
                total_events += len(events)
            
            # Simple trend analysis
            improvement_trend = "stable"
            if total_trips > 3:
                recent_scores = [t.overall_score for t in trips[-3:]]
                if recent_scores[-1] > recent_scores[0]:
                    improvement_trend = "improving"
                elif recent_scores[-1] < recent_scores[0]:
                    improvement_trend = "declining"
            
            # Generate recommendations
            recommendations = []
            events_per_trip = total_events / total_trips if total_trips > 0 else 0
            
            if events_per_trip > 3:
                recommendations.append("Focus on smoother driving to reduce harsh events")
            if average_score < 70:
                recommendations.append("Work on improving overall driving scores")
            if average_score > 85:
                recommendations.append("Excellent driving! Keep up the good work!")
            
            if not recommendations:
                recommendations.append("Continue your safe driving habits")
            
            return {
                "total_trips": total_trips,
                "average_score": round(average_score, 1),
                "improvement_trend": improvement_trend,
                "recommendations": recommendations
            }
    
    def get_risk_breakdown(self, driver_id: str) -> Dict[str, Any]:
        """Get risk score breakdown for a driver."""
        with get_session_context() as session:
            risk_repo = RiskScoreRepository()
            
            try:
                driver_uuid = UUID(driver_id)
            except ValueError:
                return {"overall_score": 75.0, "confidence": 0.0, "factors": [], "category": "medium"}
                
            risk_scores = risk_repo.get_by_driver(session, driver_uuid)
            
            if not risk_scores:
                return {
                    "overall_score": 75.0,
                    "confidence": 0.0,
                    "factors": ["No risk data available"],
                    "category": "medium"
                }
            
            latest_risk_score = max(risk_scores, key=lambda rs: rs.calculated_at)
            
            # Parse factors from JSON string
            try:
                factors = json.loads(latest_risk_score.factors) if isinstance(latest_risk_score.factors, str) else latest_risk_score.factors
            except:
                factors = ["Standard risk factors"]
            
            # Determine category
            score = latest_risk_score.score
            if score <= 30:
                category = "low"
            elif score <= 60:
                category = "medium"
            elif score <= 80:
                category = "high"
            else:
                category = "very_high"
            
            return {
                "overall_score": round(latest_risk_score.score, 1),
                "confidence": round(latest_risk_score.confidence, 2),
                "factors": factors,
                "category": category
            }