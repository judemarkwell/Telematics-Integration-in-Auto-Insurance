"""
Dashboard Service for Telematics Insurance.

This module provides services for generating dashboard data and insights
for users based on their driving behavior and risk scores.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..domain.entities import Driver, Policy, RiskScore
from ..data_processing.data_processor import TripMetrics


@dataclass
class DashboardSummary:
    """Summary data for user dashboard."""
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


class DashboardService:
    """Service for generating dashboard data and insights."""
    
    def __init__(self):
        """Initialize the dashboard service."""
        # In-memory storage for demo purposes
        self.drivers_data: Dict[str, Driver] = {}
        self.policies_data: Dict[str, Policy] = {}
        self.risk_scores_data: Dict[str, RiskScore] = {}
        self.trip_metrics_data: Dict[str, List[TripMetrics]] = {}
    
    def get_dashboard_summary(self, driver_id: str) -> Optional[DashboardSummary]:
        """
        Get dashboard summary for a driver.
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            DashboardSummary or None if driver not found
        """
        if driver_id not in self.drivers_data:
            return None
        
        driver = self.drivers_data[driver_id]
        risk_score = self.risk_scores_data.get(driver_id)
        trips = self.trip_metrics_data.get(driver_id, [])
        
        # Calculate summary data
        current_risk_score = risk_score.score if risk_score else 75.0
        risk_category = self._get_risk_category(current_risk_score)
        
        # Get current premium (mock for now)
        current_premium = 1200.0
        premium_change = 0.0
        
        total_trips = len(trips)
        total_distance = sum(trip.total_distance.value for trip in trips) if trips else 0.0
        average_score = sum(trip.overall_score for trip in trips) / len(trips) if trips else 75.0
        last_trip_date = max(trip.start_time for trip in trips) if trips else None
        
        return DashboardSummary(
            driver_name=driver.name,
            current_risk_score=current_risk_score,
            risk_category=risk_category,
            current_premium=current_premium,
            premium_change=premium_change,
            total_trips=total_trips,
            total_distance=total_distance,
            average_score=average_score,
            last_trip_date=last_trip_date,
            policy_status="active"
        )
    
    def get_recent_trips(self, driver_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trips for a driver.
        
        Args:
            driver_id: Driver identifier
            limit: Maximum number of trips to return
            
        Returns:
            List of trip summaries
        """
        trips = self.trip_metrics_data.get(driver_id, [])
        
        # Sort by start time (most recent first)
        trips.sort(key=lambda t: t.start_time, reverse=True)
        
        # Limit results
        trips = trips[:limit]
        
        # Convert to summary format
        trip_summaries = []
        for trip in trips:
            duration_minutes = int(trip.total_duration.total_seconds() / 60)
            
            trip_summaries.append({
                "trip_id": trip.trip_id,
                "date": trip.start_time,
                "duration_minutes": duration_minutes,
                "distance_km": trip.total_distance.value,
                "average_speed": trip.average_speed.value,
                "max_speed": trip.max_speed.value,
                "score": trip.overall_score,
                "events_count": (trip.hard_braking_events + 
                               trip.hard_acceleration_events + 
                               trip.speeding_events + 
                               trip.sharp_turn_events),
                "route_summary": f"{trip.total_distance.value:.1f}km in {duration_minutes}min"
            })
        
        return trip_summaries
    
    def get_driving_insights(self, driver_id: str) -> Dict[str, Any]:
        """
        Get driving behavior insights for a driver.
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            Dictionary of driving insights
        """
        trips = self.trip_metrics_data.get(driver_id, [])
        
        if not trips:
            return {
                "total_trips": 0,
                "average_score": 0,
                "improvement_trend": "no_data",
                "recommendations": ["Start driving to generate insights"]
            }
        
        # Calculate insights
        total_trips = len(trips)
        average_score = sum(trip.overall_score for trip in trips) / total_trips
        
        # Calculate improvement trend
        if total_trips >= 5:
            recent_trips = trips[-5:]
            older_trips = trips[-10:-5] if len(trips) >= 10 else trips[:-5]
            
            recent_avg = sum(trip.overall_score for trip in recent_trips) / len(recent_trips)
            older_avg = sum(trip.overall_score for trip in older_trips) / len(older_trips)
            
            if recent_avg > older_avg + 5:
                improvement_trend = "improving"
            elif recent_avg < older_avg - 5:
                improvement_trend = "declining"
            else:
                improvement_trend = "stable"
        else:
            improvement_trend = "insufficient_data"
        
        # Generate recommendations
        recommendations = []
        if average_score < 60:
            recommendations.append("Focus on smoother acceleration and braking")
            recommendations.append("Reduce speeding incidents")
        elif average_score > 80:
            recommendations.append("Excellent driving! Keep up the good work")
        
        return {
            "total_trips": total_trips,
            "average_score": average_score,
            "improvement_trend": improvement_trend,
            "recommendations": recommendations
        }
    
    def get_premium_history(self, driver_id: str, months: int = 12) -> List[Dict[str, Any]]:
        """
        Get premium history for a driver.
        
        Args:
            driver_id: Driver identifier
            months: Number of months of history to return
            
        Returns:
            List of premium history entries
        """
        # Mock premium history for demo
        history = []
        base_premium = 1200.0
        
        for i in range(months):
            date = datetime.utcnow() - timedelta(days=i * 30)
            # Simulate some variation in premium
            variation = 1.0 + (i * 0.02)  # Slight increase over time
            premium = base_premium * variation
            
            history.append({
                "date": date,
                "premium": premium,
                "change": premium - base_premium
            })
        
        return history
    
    def get_risk_breakdown(self, driver_id: str) -> Dict[str, Any]:
        """
        Get detailed risk score breakdown for a driver.
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            Dictionary of risk breakdown
        """
        risk_score = self.risk_scores_data.get(driver_id)
        
        if not risk_score:
            return {
                "overall_score": 75.0,
                "factors": {
                    "driving_behavior": 75.0,
                    "time_of_day": 75.0,
                    "road_conditions": 75.0,
                    "vehicle_safety": 75.0
                },
                "confidence": 0.5
            }
        
        return {
            "overall_score": risk_score.score,
            "factors": {
                "driving_behavior": risk_score.score * 0.8,
                "time_of_day": risk_score.score * 1.1,
                "road_conditions": risk_score.score * 0.9,
                "vehicle_safety": risk_score.score * 1.0
            },
            "confidence": risk_score.confidence
        }
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Get risk category based on score."""
        if risk_score <= 30:
            return "Low"
        elif risk_score <= 60:
            return "Medium"
        elif risk_score <= 80:
            return "High"
        else:
            return "Very High"
