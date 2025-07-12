import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Mock environment variables before importing app
os.environ.setdefault('LOCAL_DEVELOPMENT', 'true')

from app.main import app
from app.services.predictor import XGBMediaPredictor


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing."""
    predictor = MagicMock(spec=XGBMediaPredictor)
    predictor.model_name = "test-model"
    predictor.loaded_model_version = "1"
    predictor.run_id = "test-run-id"
    predictor.feature_names = ["feature1", "feature2", "feature3"]
    predictor.normalization = {"min": 0, "max": 100}
    predictor.engineered_schema = {"mapping": "test"}
    predictor.model = MagicMock()
    predictor.genres = ["Action", "Comedy", "Drama"]
    predictor.origin_countries = ["US", "UK", "CA"]
    predictor.production_countries = ["US", "UK", "CA"]
    predictor.spoken_languages = ["en", "fr", "es"]
    
    # Mock prediction methods
    predictor.predict.return_value = {
        "imdb_id": "tt0111161",
        "prediction": True,
        "probability": 0.85
    }
    
    # Mock predict_batch to handle different input scenarios
    def mock_predict_batch(items):
        if not items:  # Empty list
            return []
        
        # Generate mock results based on input items
        results = []
        for item in items:
            imdb_id = item.get("imdb_id", "unknown") if isinstance(item, dict) else item.imdb_id
            
            # Different mock results based on IMDB ID
            if imdb_id == "tt0111161":
                prediction = True
                probability = 0.85
            elif imdb_id == "tt0068646":
                prediction = False
                probability = 0.25
            else:
                prediction = True
                probability = 0.75
                
            results.append({
                "imdb_id": imdb_id,
                "prediction": prediction,
                "probability": probability
            })
        return results
    
    predictor.predict_batch.side_effect = mock_predict_batch
    
    return predictor


@pytest.fixture
def client_with_mock_predictor(mock_predictor):
    """Create a test client with mocked predictor."""
    # Mock the global predictor in main module
    import app.main
    
    # Store original predictor to restore later
    original_predictor = app.main.predictor
    
    # Set mock predictor
    app.main.predictor = mock_predictor
    
    # Mark app as being in test mode
    app.main.app.state.test_mode = True
    
    # Also ensure the predictor is passed to routers
    from app.routers import prediction
    
    try:
        with TestClient(app.main.app) as client:
            yield client
    finally:
        # Clean up: restore original predictor
        app.main.predictor = original_predictor
        app.main.app.state.test_mode = False


@pytest.fixture
def client_no_predictor():
    """Create a test client with no predictor loaded."""
    import app.main
    
    # Store original predictor to restore later
    original_predictor = app.main.predictor
    
    # Set predictor to None
    app.main.predictor = None
    
    # Mark app as being in test mode
    app.main.app.state.test_mode = True
    
    try:
        with TestClient(app.main.app) as client:
            yield client
    finally:
        # Clean up: restore original predictor
        app.main.predictor = original_predictor
        app.main.app.state.test_mode = False


@pytest.fixture
def sample_media_input():
    """Sample media input for testing."""
    return {
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


@pytest.fixture
def minimal_media_input():
    """Minimal valid media input for testing."""
    return {
        "imdb_id": "tt0111161"
    }


@pytest.fixture
def batch_media_input(sample_media_input):
    """Batch media input for testing."""
    return {
        "items": [
            sample_media_input,
            {
                "imdb_id": "tt0068646",
                "release_year": 1972,
                "genre": ["Crime", "Drama"],
                "runtime": 175
            }
        ]
    }