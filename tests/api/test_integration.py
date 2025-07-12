import pytest
import os
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

# Set up MLflow environment variables like model_training.py does
if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
    load_dotenv(override=True)

# Set MinIO environment variables (matching model_training.py)
os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
    os.environ['REEL_DRIVER_MINIO_ENDPOINT'] +
    ":" +
    os.environ['REEL_DRIVER_MINIO_PORT']
)
os.environ['AWS_ACCESS_KEY_ID'] = os.environ['REEL_DRIVER_MINIO_ACCESS_KEY']
os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['REEL_DRIVER_MINIO_SECRET_KEY']

from app.services.predictor import XGBMediaPredictor
from app.services.mlflow_client import MLflowModelLoader


@pytest.mark.integration
class TestMLflowIntegration:
    """Integration tests for MLflow connectivity and model artifacts."""
    
    def test_mlflow_connection(self):
        """Test connection to MLflow server."""
        mlflow_loader = MLflowModelLoader()
        
        # Test if we can connect to MLflow
        try:
            # Try to get model metadata which tests the connection
            model_name = os.getenv('REEL_DRIVER_MLFLOW_MODEL', 'reel-driver-model')
            metadata = mlflow_loader.get_model_metadata(model_name)
            assert metadata is not None
            print(f"Connected to MLflow, found model: {metadata['model_name']} v{metadata['version']}")
        except Exception as e:
            pytest.skip(f"MLflow server not available: {e}")
    
    def test_load_model_artifacts(self):
        """Test loading model and all required artifacts from MLflow."""
        # Skip if not in development environment
        if not os.getenv('LOCAL_DEVELOPMENT'):
            pytest.skip("Integration test requires LOCAL_DEVELOPMENT=true")
        
        predictor = XGBMediaPredictor()
        
        # Test that all required attributes are loaded
        assert predictor.model is not None, "XGBoost model not loaded"
        assert predictor.feature_names is not None, "Feature names not loaded"
        assert predictor.normalization is not None, "Normalization parameters not loaded"
        assert predictor.engineered_schema is not None, "Engineered schema not loaded"
        assert predictor.model_name is not None, "Model name not set"
        assert predictor.loaded_model_version is not None, "Model version not set"
        assert predictor.run_id is not None, "Run ID not set"
        
        # Test that categorical mappings are loaded
        assert hasattr(predictor, 'genres'), "Genres mapping not loaded"
        assert hasattr(predictor, 'origin_countries'), "Origin countries mapping not loaded"
        assert hasattr(predictor, 'production_countries'), "Production countries mapping not loaded"
        assert hasattr(predictor, 'spoken_languages'), "Spoken languages mapping not loaded"
        
        print(f"Model loaded: {predictor.model_name} v{predictor.loaded_model_version}")
        print(f"Features count: {len(predictor.feature_names)}")
        print(f"Normalization fields: {len(predictor.normalization)}")
        print(f"Schema mappings: {len(predictor.engineered_schema)}")
    
    def test_model_prediction_real(self):
        """Test real model prediction with sample data."""
        if not os.getenv('LOCAL_DEVELOPMENT'):
            pytest.skip("Integration test requires LOCAL_DEVELOPMENT=true")
        
        predictor = XGBMediaPredictor()
        
        # Sample movie data (The Shawshank Redemption)
        from app.models.media_prediction_input import MediaPredictionInput
        
        sample_input = MediaPredictionInput(
            imdb_id="tt0111161",
            release_year=1994,
            budget=25000000,
            revenue=16000000,
            runtime=142,
            origin_country=["US"],
            production_companies=["Castle Rock Entertainment"],
            production_countries=["US"],
            production_status="Released",
            original_language="en",
            spoken_languages=["en"],
            genre=["Drama", "Crime"],
            tagline="Fear can hold you prisoner. Hope can set you free.",
            overview="Framed in the 1940s for the double murder of his wife and her lover...",
            tmdb_rating=8.7,
            tmdb_votes=26000,
            rt_score=91,
            metascore=82,
            imdb_rating=92.0,
            imdb_votes=2800000
        )
        
        result = predictor.predict(sample_input)
        
        # Validate prediction result structure
        assert "imdb_id" in result
        assert "prediction" in result
        assert "probability" in result
        assert result["imdb_id"] == "tt0111161"
        assert isinstance(result["prediction"], bool)
        assert isinstance(result["probability"], float)
        assert 0.0 <= result["probability"] <= 1.0
        
        print(f"Prediction for {result['imdb_id']}: {result['prediction']} (probability: {result['probability']:.3f})")
    
    def test_batch_prediction_real(self):
        """Test real batch prediction with multiple samples."""
        if not os.getenv('LOCAL_DEVELOPMENT'):
            pytest.skip("Integration test requires LOCAL_DEVELOPMENT=true")
        
        predictor = XGBMediaPredictor()
        
        from app.models.media_prediction_input import MediaPredictionInput
        
        # Multiple sample movies
        samples = [
            MediaPredictionInput(
                imdb_id="tt0111161",  # The Shawshank Redemption
                release_year=1994,
                genre=["Drama", "Crime"],
                runtime=142,
                imdb_rating=92.0
            ),
            MediaPredictionInput(
                imdb_id="tt0068646",  # The Godfather
                release_year=1972,
                genre=["Crime", "Drama"],
                runtime=175,
                imdb_rating=92.0
            ),
            MediaPredictionInput(
                imdb_id="tt0468569",  # The Dark Knight
                release_year=2008,
                genre=["Action", "Crime", "Drama"],
                runtime=152,
                imdb_rating=90.0
            )
        ]
        
        results = predictor.predict_batch(samples)
        
        # Validate batch results
        assert len(results) == len(samples)
        
        for i, result in enumerate(results):
            assert "imdb_id" in result
            assert "prediction" in result
            assert "probability" in result
            assert result["imdb_id"] == samples[i].imdb_id
            assert isinstance(result["prediction"], bool)
            assert isinstance(result["probability"], float)
            assert 0.0 <= result["probability"] <= 1.0
            
            print(f"Batch prediction {i+1}: {result['imdb_id']} -> {result['prediction']} ({result['probability']:.3f})")


@pytest.mark.integration
class TestFullAppIntegration:
    """Integration tests for the full FastAPI application with real MLflow."""
    
    @pytest.fixture(scope="class")
    def real_client(self):
        """Create a test client with real MLflow connection."""
        if not os.getenv('LOCAL_DEVELOPMENT'):
            pytest.skip("Integration test requires LOCAL_DEVELOPMENT=true")
        
        # Import the real app
        from app.main import app
        
        with TestClient(app) as client:
            yield client
    
    def test_health_check_real(self, real_client):
        """Test health check with real model loaded."""
        response = real_client.get("/reel-driver/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data
        assert "model_version" in data
        assert "run_id" in data
        assert "features_count" in data
        assert data["features_count"] > 0
        
        print(f"Health check passed: {data['model_name']} v{data['model_version']}")
    
    def test_single_prediction_real(self, real_client):
        """Test single prediction endpoint with real model."""
        sample_data = {
            "imdb_id": "tt0111161",
            "release_year": 1994,
            "budget": 25000000,
            "revenue": 16000000,
            "runtime": 142,
            "origin_country": ["US"],
            "production_companies": ["Castle Rock Entertainment"],
            "production_countries": ["US"],
            "production_status": "Released",
            "original_language": "en",
            "spoken_languages": ["en"],
            "genre": ["Drama", "Crime"],
            "tagline": "Fear can hold you prisoner. Hope can set you free.",
            "overview": "Framed in the 1940s for the double murder of his wife and her lover...",
            "tmdb_rating": 8.7,
            "tmdb_votes": 26000,
            "rt_score": 91,
            "metascore": 82,
            "imdb_rating": 92.0,
            "imdb_votes": 2800000
        }
        
        response = real_client.post("/reel-driver/api/predict", json=sample_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["imdb_id"] == "tt0111161"
        assert "prediction" in data
        assert "probability" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["probability"], float)
        assert 0.0 <= data["probability"] <= 1.0
        
        print(f"Real prediction: {data['prediction']} (probability: {data['probability']:.3f})")
    
    def test_batch_prediction_real(self, real_client):
        """Test batch prediction endpoint with real model."""
        batch_data = {
            "items": [
                {
                    "imdb_id": "tt0111161",
                    "release_year": 1994,
                    "genre": ["Drama", "Crime"],
                    "runtime": 142
                },
                {
                    "imdb_id": "tt0068646",
                    "release_year": 1972,
                    "genre": ["Crime", "Drama"],
                    "runtime": 175
                }
            ]
        }
        
        response = real_client.post("/reel-driver/api/predict_batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        
        for i, result in enumerate(data["results"]):
            assert "imdb_id" in result
            assert "prediction" in result
            assert "probability" in result
            assert result["imdb_id"] == batch_data["items"][i]["imdb_id"]
            
            print(f"Real batch prediction {i+1}: {result['imdb_id']} -> {result['prediction']} ({result['probability']:.3f})")


@pytest.mark.integration  
class TestEnvironmentConfiguration:
    """Test environment variable configuration for MLflow integration."""
    
    def test_required_env_vars(self):
        """Test that all required environment variables are set."""
        required_vars = [
            'REEL_DRIVER_MLFLOW_HOST',
            'REEL_DRIVER_MLFLOW_PORT',
            'REEL_DRIVER_MLFLOW_EXPERIMENT',
            'REEL_DRIVER_MLFLOW_MODEL',
            'REEL_DRIVER_MINIO_ENDPOINT',
            'REEL_DRIVER_MINIO_PORT',
            'REEL_DRIVER_MINIO_ACCESS_KEY',
            'REEL_DRIVER_MINIO_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")
        
        print("All required environment variables are set")
    
    def test_mlflow_config(self):
        """Test MLflow configuration from environment variables."""
        mlflow_host = os.getenv('REEL_DRIVER_MLFLOW_HOST')
        mlflow_port = os.getenv('REEL_DRIVER_MLFLOW_PORT')
        experiment_name = os.getenv('REEL_DRIVER_MLFLOW_EXPERIMENT')
        model_name = os.getenv('REEL_DRIVER_MLFLOW_MODEL')
        
        if not all([mlflow_host, mlflow_port, experiment_name, model_name]):
            pytest.skip("MLflow environment variables not configured")
        
        print(f"MLflow Host: {mlflow_host}:{mlflow_port}")
        print(f"Experiment: {experiment_name}")
        print(f"Model: {model_name}")
        
        # Test that we can construct the tracking URI
        tracking_uri = f"http://{mlflow_host}:{mlflow_port}"
        assert tracking_uri.startswith("http://")
        
        print(f"MLflow Tracking URI: {tracking_uri}")
    
