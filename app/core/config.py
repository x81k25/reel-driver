from pydantic import Field
from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    REEL_DRIVER_API_HOST: str = Field(default="0.0.0.0", description="Host to bind the API server")
    REEL_DRIVER_API_PORT: int = Field(default=8000, description="Port to run the API server")

    # MLflow Configuration (matching training script)
    REEL_DRIVER_MLFLOW_HOST: str = Field(description="MLflow tracking server host")
    REEL_DRIVER_MLFLOW_PORT: str = Field(description="MLflow tracking server port")
    REEL_DRIVER_MLFLOW_EXPERIMENT: str = Field(description="MLflow experiment name")
    REEL_DRIVER_MLFLOW_MODEL: str = Field(description="MLflow registered model name")
    
    # MinIO Configuration (matching training script)
    REEL_DRIVER_MINIO_ENDPOINT: str = Field(description="MinIO endpoint URL")
    REEL_DRIVER_MINIO_PORT: str = Field(description="MinIO port")
    REEL_DRIVER_MINIO_ACCESS_KEY: str = Field(description="MinIO access key")
    REEL_DRIVER_MINIO_SECRET_KEY: str = Field(description="MinIO secret key")

    # Model version (default to latest)
    REEL_DRIVER_MODEL_VERSION: Optional[str] = Field(default="latest", description="MLflow model version")

    # Legacy Model Configuration (deprecated)
    MODEL_PATH: str = Field(default="./model_artifacts/", description="Path to model artifacts")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    class Config:
        env_file = ".env"  # Use root .env file
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings()