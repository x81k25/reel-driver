from pydantic import Field
from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="Host to bind the API server")
    API_PORT: int = Field(default=8000, description="Port to run the API server")

    # Model Configuration
    MODEL_PATH: str = Field(default="./model_artifacts/", description="Path to model artifacts")

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Database Configuration
    PG_DB: Optional[str] = None
    PG_USER: Optional[str] = None
    PG_PASS: Optional[str] = None
    PG_HOST: Optional[str] = None
    PG_PORT: Optional[str] = None
    PG_SCHEMA: Optional[str] = None

    # MLflow Configuration
    MLFLOW_HOST: Optional[str] = None
    MLFLOW_PORT: Optional[str] = None

    # AWS Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    MLFLOW_S3_ENDPOINT_URL: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()