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

    class Config:
        env_file = "./app/.env"
        case_sensitive = True

# Create settings instance
settings = Settings()