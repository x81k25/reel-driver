# standard library imports
import json
import logging
import os
import tempfile
from typing import Dict, Any, Optional

# third party imports
import mlflow
from mlflow.tracking import MlflowClient

# custom/local imports
from app.core.config import settings

logger = logging.getLogger("uvicorn.error")


class MLflowModelLoader:
    """
    Service for loading models and artifacts from MLflow.
    Handles MLflow client configuration and artifact download.
    """

    def __init__(self):
        """Initialize MLflow client with configuration."""
        self._setup_mlflow()
        self.client = MlflowClient()

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

    def get_latest_model_version(self, model_name: str) -> str:
        """
        Get the latest version number for a registered model.
        
        :param model_name: Name of the registered model
        :return: Latest version number as string
        """
        try:
            latest_version = self.client.get_latest_versions(
                model_name, 
                stages=["Production", "Staging", "None"]
            )[0]
            logger.info(f"Latest model version for {model_name}: {latest_version.version}")
            return latest_version.version
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            raise

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
            model_path = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded successfully")
            
            # Get the run ID to download artifacts
            model_version = self.client.get_model_version(model_name, version)
            run_id = model_version.run_id
            
            # Download normalization table artifact to temp directory
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
            
            # Download engineered schema artifact
            schema_artifact_path = self.client.download_artifacts(
                run_id, "model-artifacts/engineered_schema.json", temp_dir
            )
            
            with open(schema_artifact_path, 'r') as f:
                schema_data = json.load(f)
            
            logger.info("All artifacts downloaded successfully")
            
            return {
                'model': model_path,
                'normalization': normalization_dict,
                'schema': schema_data,
                'run_id': run_id,
                'model_version': version
            }
            
        except Exception as e:
            logger.error(f"Failed to download model artifacts: {e}")
            raise

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
            
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise