"""
Application configuration settings.

This module centralizes all configuration parameters for the telematics
insurance system, following the principle of configuration as code.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration parameters."""
    host: str
    port: int
    name: str
    user: str
    password: str
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create database config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "telematics_insurance"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password")
        )


@dataclass(frozen=True)
class APIConfig:
    """API configuration parameters."""
    host: str
    port: int
    debug: bool
    secret_key: str
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create API config from environment variables."""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "False").lower() == "true",
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-here")
        )


@dataclass(frozen=True)
class MLConfig:
    """Machine learning model configuration."""
    model_path: str
    retrain_interval_hours: int
    feature_importance_threshold: float
    
    @classmethod
    def from_env(cls) -> "MLConfig":
        """Create ML config from environment variables."""
        return cls(
            model_path=os.getenv("MODEL_PATH", "models/"),
            retrain_interval_hours=int(os.getenv("RETRAIN_INTERVAL_HOURS", "24")),
            feature_importance_threshold=float(os.getenv("FEATURE_THRESHOLD", "0.1"))
        )


@dataclass(frozen=True)
class TelematicsConfig:
    """Telematics data collection configuration."""
    sampling_rate_hz: int
    gps_accuracy_threshold: float
    max_speed_kmh: float
    
    @classmethod
    def from_env(cls) -> "TelematicsConfig":
        """Create telematics config from environment variables."""
        return cls(
            sampling_rate_hz=int(os.getenv("SAMPLING_RATE_HZ", "10")),
            gps_accuracy_threshold=float(os.getenv("GPS_ACCURACY_THRESHOLD", "5.0")),
            max_speed_kmh=float(os.getenv("MAX_SPEED_KMH", "200.0"))
        )


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig
    api: APIConfig
    ml: MLConfig
    telematics: TelematicsConfig
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create complete app config from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            api=APIConfig.from_env(),
            ml=MLConfig.from_env(),
            telematics=TelematicsConfig.from_env()
        )


# Global configuration instance
config = AppConfig.from_env()

