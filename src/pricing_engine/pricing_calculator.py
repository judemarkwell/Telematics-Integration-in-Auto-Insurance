"""
Pricing Calculator for Telematics Insurance.

This module provides pricing calculation services that adjust insurance
premiums based on risk scores, driving behavior, and other factors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import math

from ..domain.value_objects import Premium
from ..data_processing.data_processor import TripMetrics


class RiskCategory(Enum):
    """Risk categories for pricing calculations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PricingFactors:
    """Factors that influence insurance pricing."""
    base_premium: float
    risk_score: float
    driving_experience_years: int = 5
    vehicle_safety_rating: float = 4.0
    vehicle_age: int = 3
    annual_mileage: float = 12000.0
    location_risk_factor: float = 1.0
    claims_history: int = 0
    time_since_last_claim: int = 24
    credit_score: int = 700
    age: int = 30
    marital_status: str = "single"
    education_level: str = "college"
    occupation: str = "professional"
    policy_duration_months: int = 12


@dataclass
class PricingResult:
    """Result of pricing calculation."""
    base_premium: float
    adjusted_premium: float
    discount_amount: float
    discount_percentage: float
    risk_category: RiskCategory
    factors_applied: Dict[str, float]
    calculation_date: datetime
    confidence_score: float


class PricingService:
    """Service for calculating insurance premiums based on telematics data."""
    
    def __init__(self):
        """Initialize the pricing service."""
        self.risk_thresholds = {
            RiskCategory.LOW: 30,      # 0-30: Low risk
            RiskCategory.MEDIUM: 60,   # 30-60: Medium risk  
            RiskCategory.HIGH: 80,     # 60-80: High risk
            RiskCategory.VERY_HIGH: 100 # 80-100: Very high risk
        }
        
        # Pricing adjustment factors
        self.risk_adjustments = {
            RiskCategory.LOW: 0.8,      # 20% discount
            RiskCategory.MEDIUM: 1.0,   # No change
            RiskCategory.HIGH: 1.3,     # 30% increase
            RiskCategory.VERY_HIGH: 1.6 # 60% increase
        }
    
    def calculate_premium(self, factors: PricingFactors) -> PricingResult:
        """
        Calculate adjusted premium based on pricing factors.
        
        Args:
            factors: Pricing factors including risk score and other data
            
        Returns:
            PricingResult with adjusted premium and applied factors
        """
        base_premium = factors.base_premium
        risk_score = factors.risk_score
        
        # Determine risk category
        risk_category = self._get_risk_category(risk_score)
        
        # Calculate adjustments
        adjustments = self._calculate_adjustments(factors)
        
        # Apply risk-based adjustment
        risk_adjustment = self.risk_adjustments[risk_category]
        
        # Calculate final premium
        total_adjustment = 1.0
        for factor_name, adjustment in adjustments.items():
            total_adjustment *= adjustment
        
        total_adjustment *= risk_adjustment
        
        adjusted_premium = base_premium * total_adjustment
        discount_amount = base_premium - adjusted_premium
        discount_percentage = (discount_amount / base_premium) * 100
        
        # Calculate confidence score based on data quality
        confidence_score = self._calculate_confidence_score(factors)
        
        return PricingResult(
            base_premium=base_premium,
            adjusted_premium=adjusted_premium,
            discount_amount=discount_amount,
            discount_percentage=discount_percentage,
            risk_category=risk_category,
            factors_applied=adjustments,
            calculation_date=datetime.utcnow(),
            confidence_score=confidence_score
        )
    
    def _get_risk_category(self, risk_score: float) -> RiskCategory:
        """Determine risk category based on risk score."""
        if risk_score <= self.risk_thresholds[RiskCategory.LOW]:
            return RiskCategory.LOW
        elif risk_score <= self.risk_thresholds[RiskCategory.MEDIUM]:
            return RiskCategory.MEDIUM
        elif risk_score <= self.risk_thresholds[RiskCategory.HIGH]:
            return RiskCategory.HIGH
        else:
            return RiskCategory.VERY_HIGH
    
    def _calculate_adjustments(self, factors: PricingFactors) -> Dict[str, float]:
        """Calculate individual adjustment factors."""
        adjustments = {}
        
        # Driving experience adjustment
        if factors.driving_experience_years >= 10:
            adjustments["driving_experience"] = 0.95  # 5% discount
        elif factors.driving_experience_years >= 5:
            adjustments["driving_experience"] = 1.0   # No change
        else:
            adjustments["driving_experience"] = 1.15  # 15% increase
        
        # Vehicle safety rating adjustment
        if factors.vehicle_safety_rating >= 4.5:
            adjustments["vehicle_safety"] = 0.9   # 10% discount
        elif factors.vehicle_safety_rating >= 4.0:
            adjustments["vehicle_safety"] = 1.0   # No change
        else:
            adjustments["vehicle_safety"] = 1.1   # 10% increase
        
        # Vehicle age adjustment
        if factors.vehicle_age <= 2:
            adjustments["vehicle_age"] = 1.0   # No change for new vehicles
        elif factors.vehicle_age <= 5:
            adjustments["vehicle_age"] = 1.0   # No change for newer vehicles
        else:
            adjustments["vehicle_age"] = 1.05  # 5% increase for older vehicles
        
        # Annual mileage adjustment
        if factors.annual_mileage <= 8000:
            adjustments["mileage"] = 0.9   # 10% discount
        elif factors.annual_mileage <= 15000:
            adjustments["mileage"] = 1.0   # No change
        else:
            adjustments["mileage"] = 1.1   # 10% increase
        
        # Location risk adjustment
        adjustments["location"] = factors.location_risk_factor
        
        # Claims history adjustment
        if factors.claims_history == 0:
            adjustments["claims_history"] = 0.95  # 5% discount
        elif factors.claims_history == 1:
            adjustments["claims_history"] = 1.0   # No change
        else:
            adjustments["claims_history"] = 1.2   # 20% increase
        
        # Time since last claim adjustment
        if factors.time_since_last_claim >= 36:
            adjustments["time_since_claim"] = 0.95  # 5% discount
        elif factors.time_since_last_claim >= 24:
            adjustments["time_since_claim"] = 1.0   # No change
        else:
            adjustments["time_since_claim"] = 1.1   # 10% increase
        
        # Credit score adjustment
        if factors.credit_score >= 750:
            adjustments["credit_score"] = 0.95  # 5% discount
        elif factors.credit_score >= 700:
            adjustments["credit_score"] = 1.0   # No change
        else:
            adjustments["credit_score"] = 1.1   # 10% increase
        
        # Age adjustment
        if 25 <= factors.age <= 65:
            adjustments["age"] = 1.0   # No change
        elif factors.age < 25:
            adjustments["age"] = 1.2   # 20% increase for young drivers
        else:
            adjustments["age"] = 1.1   # 10% increase for older drivers
        
        # Marital status adjustment
        if factors.marital_status == "married":
            adjustments["marital_status"] = 0.95  # 5% discount
        else:
            adjustments["marital_status"] = 1.0   # No change
        
        return adjustments
    
    def _calculate_confidence_score(self, factors: PricingFactors) -> float:
        """Calculate confidence score based on data quality and completeness."""
        confidence = 1.0
        
        # Reduce confidence for missing or incomplete data
        if factors.risk_score < 0 or factors.risk_score > 1:
            confidence -= 0.2
        
        if factors.driving_experience_years < 0:
            confidence -= 0.1
        
        if factors.vehicle_safety_rating < 1 or factors.vehicle_safety_rating > 5:
            confidence -= 0.1
        
        if factors.annual_mileage < 0:
            confidence -= 0.1
        
        if factors.claims_history < 0:
            confidence -= 0.1
        
        if factors.credit_score < 300 or factors.credit_score > 850:
            confidence -= 0.1
        
        if factors.age < 16 or factors.age > 100:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def calculate_premium_from_trip_metrics(
        self, 
        base_premium: float, 
        trip_metrics: List[TripMetrics],
        additional_factors: Optional[Dict[str, Any]] = None
    ) -> PricingResult:
        """
        Calculate premium based on trip metrics and additional factors.
        
        Args:
            base_premium: Base insurance premium
            trip_metrics: List of trip metrics for risk assessment
            additional_factors: Additional pricing factors
            
        Returns:
            PricingResult with calculated premium
        """
        if not trip_metrics:
            # Default factors if no trip data available
            factors = PricingFactors(
                base_premium=base_premium,
                risk_score=0.5,  # Default medium risk
                **(additional_factors or {})
            )
        else:
            # Calculate average risk score from trip metrics
            total_score = sum(trip.overall_score for trip in trip_metrics)
            avg_risk_score = total_score / len(trip_metrics)
            
            # Normalize to 0-1 range
            normalized_risk_score = min(1.0, max(0.0, avg_risk_score / 100.0))
            
            factors = PricingFactors(
                base_premium=base_premium,
                risk_score=normalized_risk_score,
                **(additional_factors or {})
            )
        
        return self.calculate_premium(factors)
    
    def get_pricing_recommendations(self, result: PricingResult) -> List[str]:
        """
        Get recommendations for improving pricing.
        
        Args:
            result: Pricing calculation result
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if result.risk_category == RiskCategory.HIGH or result.risk_category == RiskCategory.VERY_HIGH:
            recommendations.append("Consider taking a defensive driving course to improve your risk score")
            recommendations.append("Reduce high-risk driving behaviors like hard braking and acceleration")
            recommendations.append("Drive during safer times of day when possible")
        
        if result.factors_applied.get("driving_experience", 1.0) > 1.0:
            recommendations.append("Gain more driving experience to qualify for better rates")
        
        if result.factors_applied.get("vehicle_safety", 1.0) > 1.0:
            recommendations.append("Consider upgrading to a vehicle with better safety ratings")
        
        if result.factors_applied.get("mileage", 1.0) > 1.0:
            recommendations.append("Reduce annual mileage to qualify for lower rates")
        
        if result.factors_applied.get("claims_history", 1.0) > 1.0:
            recommendations.append("Maintain a clean driving record to improve your rates")
        
        if result.factors_applied.get("credit_score", 1.0) > 1.0:
            recommendations.append("Improve your credit score to qualify for better insurance rates")
        
        return recommendations
