from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
import os
from typing import Optional
from loguru import logger

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MLflow Configuration
    REEL_DRIVER_MLFLOW_HOST: str = Field(description="MLflow tracking server host")
    REEL_DRIVER_MLFLOW_PORT: str = Field(description="MLflow tracking server port")
    REEL_DRIVER_MLFLOW_EXPERIMENT: str = Field(description="MLflow experiment name")
    REEL_DRIVER_MLFLOW_MODEL: str = Field(description="MLflow registered model name")
    
    # MinIO Configuration
    REEL_DRIVER_MINIO_ENDPOINT: str = Field(description="MinIO endpoint URL")
    REEL_DRIVER_MINIO_PORT: str = Field(description="MinIO port")
    REEL_DRIVER_MINIO_ACCESS_KEY: str = Field(description="MinIO access key")
    REEL_DRIVER_MINIO_SECRET_KEY: str = Field(description="MinIO secret key")

    # FastAPI Configuration
    REEL_DRIVER_API_HOST: str = Field(default="0.0.0.0", description="Host to bind the API server")
    REEL_DRIVER_API_PORT: int = Field(default=8000, description="Port to run the API server")
    REEL_DRIVER_API_PREFIX: str = Field(default="reel-driver", description="API prefix")
    REEL_DRIVER_API_LOG_LEVEL: str = Field(default="INFO", description="API logging level")
    REEL_DRIVER_API_MODEL_VERSION: str = Field(default="latest", description="MLflow model version")

    class Config:
        env_file = ".env"  # Use root .env file
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance with error logging
try:
    settings = Settings()
    logger.success("All required environment variables loaded successfully")
except ValidationError as e:
    logger.error("Failed to load required environment variables:")
    for error in e.errors():
        field_name = error['loc'][0]
        error_msg = error['msg']
        logger.error(f"  {field_name}: {error_msg}")
        
        # Show if the env var exists but has wrong type/format
        env_value = os.getenv(field_name)
        if env_value is not None:
            sensitive_keywords = ["SECRET", "KEY", "PASSWORD", "TOKEN"]
            if any(keyword in field_name for keyword in sensitive_keywords):
                masked_value = env_value[:4] + "*" * (len(env_value) - 4) if len(env_value) > 4 else "****"
                logger.error(f"    Current value: {masked_value}")
            else:
                logger.error(f"    Current value: {env_value}")
        else:
            logger.error(f"    Environment variable not set")
    
    logger.error("Application cannot start without required environment variables")
    raise