# standard library imports
import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional

# third party imports
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import requests

# custom/local imports
from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class MLflowConnectionError(Exception):
    """Raised when MLflow server is unreachable."""
    pass


class MLflowModelNotFoundError(Exception):
    """Raised when requested model is not found in MLflow."""
    pass


class MLflowArtifactError(Exception):
    """Raised when model artifacts cannot be downloaded."""
    pass


class MLflowModelLoader:
    """
    Service for loading models and artifacts from MLflow.
    Handles MLflow client configuration and artifact download.
    """

    def __init__(self):
        """Initialize MLflow client with configuration."""
        try:
            self._setup_mlflow()
            self.client = MlflowClient()
            self._validate_mlflow_connection()
        except Exception as e:
            logger.error(f"Failed to initialize MLflow client: {e}")
            raise MLflowConnectionError(f"Cannot connect to MLflow server: {e}")

    def _setup_mlflow(self):
        """Configure MLflow tracking URI and S3 credentials (matching training script)."""
        # Set MinIO environment variables (matching training script)
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
            settings.REEL_DRIVER_MINIO_ENDPOINT +
            ":" +
            settings.REEL_DRIVER_MINIO_PORT
        )
        os.environ['AWS_ACCESS_KEY_ID'] = settings.REEL_DRIVER_MINIO_ACCESS_KEY
        os.environ['AWS_SECRET_ACCESS_KEY'] = settings.REEL_DRIVER_MINIO_SECRET_KEY
        
        # Set MLflow tracking URI (matching training script)
        mlflow_uri = "http://" + settings.REEL_DRIVER_MLFLOW_HOST + ":" + settings.REEL_DRIVER_MLFLOW_PORT
        mlflow.set_tracking_uri(mlflow_uri)
        
        logger.info(f"MLflow client configured with tracking URI: {mlflow_uri}")
        logger.info(f"MinIO S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
    
    def _validate_mlflow_connection(self):
        """Validate that MLflow server is reachable."""
        try:
            # Test basic connectivity to MLflow server
            mlflow_uri = "http://" + settings.REEL_DRIVER_MLFLOW_HOST + ":" + settings.REEL_DRIVER_MLFLOW_PORT
            response = requests.get(f"{mlflow_uri}/health", timeout=10)
            if response.status_code != 200:
                raise MLflowConnectionError(f"MLflow server health check failed with status {response.status_code}")
            
            # Test client connectivity
            self.client.search_experiments(max_results=1)
            logger.info("MLflow connection validated successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"MLflow server unreachable: {e}")
            raise MLflowConnectionError(f"MLflow server unreachable: {e}")
        except MlflowException as e:
            logger.error(f"MLflow client validation failed: {e}")
            raise MLflowConnectionError(f"MLflow client validation failed: {e}")

    def get_latest_model_version(self, model_name: str) -> str:
        """
        Get the latest version number for a registered model.
        
        :param model_name: Name of the registered model
        :return: Latest version number as string
        """
        try:
            versions = self.client.get_latest_versions(
                model_name, 
                stages=["Production", "Staging", "None"]
            )
            if not versions:
                raise MLflowModelNotFoundError(f"No versions found for model '{model_name}'")
            
            latest_version = versions[0]
            logger.info(f"Latest model version for {model_name}: {latest_version.version}")
            return latest_version.version
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.error(f"Model '{model_name}' not found in MLflow registry")
                raise MLflowModelNotFoundError(f"Model '{model_name}' not found in MLflow registry")
            else:
                logger.error(f"MLflow error getting latest model version: {e}")
                raise MLflowConnectionError(f"MLflow error: {e}")
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            raise MLflowConnectionError(f"Failed to get latest model version: {e}")

    def download_model_artifacts(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Download model and all artifacts from MLflow.
        
        :param model_name: Name of the registered model
        :param version: Model version (defaults to latest)
        :return: Dictionary containing model path and artifact data
        """
        if version is None or version == "latest":
            version = self.get_latest_model_version(model_name)
        
        model_uri = f"models:/{model_name}/{version}"
        logger.info(f"Loading model from URI: {model_uri}")
        
        try:
            # Create temporary directory for artifacts
            temp_dir = tempfile.mkdtemp()
            
            # Download the model
            try:
                model_path = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Model loaded successfully")
            except MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    raise MLflowModelNotFoundError(f"Model '{model_name}' version '{version}' not found")
                else:
                    raise MLflowArtifactError(f"Failed to load model: {e}")
            
            # Get the run ID to download artifacts
            try:
                model_version = self.client.get_model_version(model_name, version)
                run_id = model_version.run_id
            except MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    raise MLflowModelNotFoundError(f"Model version '{version}' not found for model '{model_name}'")
                else:
                    raise MLflowConnectionError(f"Failed to get model version info: {e}")
            
            # Download normalization table artifact to temp directory
            try:
                normalization_artifact_path = self.client.download_artifacts(
                    run_id, "model-artifacts/engineered_normalization_table.json", temp_dir
                )
                
                with open(normalization_artifact_path, 'r') as f:
                    normalization_data = json.load(f)
                
                # Convert normalization data to the expected format
                normalization_dict = {}
                for item in normalization_data:
                    normalization_dict[item['feature']] = {
                        'min': item['min'],
                        'max': item['max']
                    }
            except (MlflowException, FileNotFoundError, json.JSONDecodeError) as e:
                raise MLflowArtifactError(f"Failed to download normalization artifacts: {e}")
            
            # Download engineered schema artifact
            try:
                schema_artifact_path = self.client.download_artifacts(
                    run_id, "model-artifacts/engineered_schema.json", temp_dir
                )
                
                with open(schema_artifact_path, 'r') as f:
                    schema_data = json.load(f)
            except (MlflowException, FileNotFoundError, json.JSONDecodeError) as e:
                raise MLflowArtifactError(f"Failed to download schema artifacts: {e}")
            
            logger.info("All artifacts downloaded successfully")
            
            return {
                'model': model_path,
                'normalization': normalization_dict,
                'schema': schema_data,
                'run_id': run_id,
                'model_version': version
            }
            
        except (MLflowConnectionError, MLflowModelNotFoundError, MLflowArtifactError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading model artifacts: {e}")
            raise MLflowArtifactError(f"Unexpected error downloading model artifacts: {e}")

    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata about the model without downloading artifacts.
        
        :param model_name: Name of the registered model
        :param version: Model version (defaults to latest)
        :return: Dictionary containing model metadata
        """
        if version is None or version == "latest":
            version = self.get_latest_model_version(model_name)
        
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            return {
                'model_name': model_name,
                'version': version,
                'run_id': model_version.run_id,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'status': model_version.status,
                'description': model_version.description,
                'tags': model_version.tags,
                'run_metrics': run.data.metrics,
                'run_params': run.data.params
            }
            
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.error(f"Model '{model_name}' version '{version}' not found")
                raise MLflowModelNotFoundError(f"Model '{model_name}' version '{version}' not found")
            else:
                logger.error(f"MLflow error getting model metadata: {e}")
                raise MLflowConnectionError(f"MLflow error: {e}")
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise MLflowConnectionError(f"Failed to get model metadata: {e}")