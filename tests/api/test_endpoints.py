import pytest
from fastapi import status


class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint(self, client_with_mock_predictor):
        """Test GET / endpoint."""
        response = client_with_mock_predictor.get("/reel-driver/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Welcome to Reel Driver API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        assert "single_prediction" in data["endpoints"]
        assert "batch_prediction" in data["endpoints"]


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check_with_loaded_model(self, client_with_mock_predictor):
        """Test health check when model is properly loaded."""
        response = client_with_mock_predictor.get("/reel-driver/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_name"] == "test-model"
        assert data["model_version"] == "1"
        assert data["run_id"] == "test-run-id"
        assert data["features_count"] == 3
        assert data["genre_categories"] == 3
        assert data["origin_countries"] == 3
        assert data["production_countries"] == 3
        assert data["spoken_languages"] == 3
    
    def test_health_check_no_model(self, client_no_predictor):
        """Test health check when no model is loaded."""
        response = client_no_predictor.get("/reel-driver/health")
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["message"] == "Model not loaded"


class TestPredictionEndpoint:
    """Test the single prediction endpoint."""
    
    def test_single_prediction_success(self, client_with_mock_predictor, sample_media_input):
        """Test successful single prediction."""
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=sample_media_input
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["imdb_id"] == "tt0111161"
        assert data["prediction"] is True
        assert data["probability"] == 0.85
    
    def test_single_prediction_minimal_input(self, client_with_mock_predictor, minimal_media_input):
        """Test single prediction with minimal input."""
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=minimal_media_input
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["imdb_id"] == "tt0111161"
        assert "prediction" in data
        assert "probability" in data
    
    def test_single_prediction_invalid_imdb_id(self, client_with_mock_predictor):
        """Test single prediction with invalid IMDB ID format."""
        invalid_input = {
            "imdb_id": "invalid_id"
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_single_prediction_invalid_year(self, client_with_mock_predictor):
        """Test single prediction with invalid release year."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "release_year": 1800  # Below minimum
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_single_prediction_invalid_rating(self, client_with_mock_predictor):
        """Test single prediction with invalid rating."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "tmdb_rating": 15.0  # Above maximum
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_single_prediction_no_model(self, client_no_predictor, sample_media_input):
        """Test single prediction when no model is loaded."""
        response = client_no_predictor.post(
            "/reel-driver/api/predict",
            json=sample_media_input
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["message"] == "Model not loaded"
    
    def test_single_prediction_model_error(self, client_with_mock_predictor, sample_media_input, mock_predictor):
        """Test single prediction when model throws error."""
        mock_predictor.predict.side_effect = Exception("Model prediction failed")
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=sample_media_input
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Prediction error" in data["message"]


class TestBatchPredictionEndpoint:
    """Test the batch prediction endpoint."""
    
    def test_batch_prediction_success(self, client_with_mock_predictor, batch_media_input):
        """Test successful batch prediction."""
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict_batch",
            json=batch_media_input
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["imdb_id"] == "tt0111161"
        assert data["results"][0]["prediction"] is True
        assert data["results"][1]["imdb_id"] == "tt0068646"
        assert data["results"][1]["prediction"] is False
    
    def test_batch_prediction_empty_list(self, client_with_mock_predictor):
        """Test batch prediction with empty items list."""
        empty_batch = {"items": []}
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict_batch",
            json=empty_batch
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["results"] == []
    
    def test_batch_prediction_invalid_item(self, client_with_mock_predictor):
        """Test batch prediction with invalid item in batch."""
        invalid_batch = {
            "items": [
                {"imdb_id": "invalid_id"}
            ]
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict_batch",
            json=invalid_batch
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_batch_prediction_no_model(self, client_no_predictor, batch_media_input):
        """Test batch prediction when no model is loaded."""
        response = client_no_predictor.post(
            "/reel-driver/api/predict_batch",
            json=batch_media_input
        )
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["message"] == "Model not loaded"
    
    def test_batch_prediction_model_error(self, client_with_mock_predictor, batch_media_input, mock_predictor):
        """Test batch prediction when model throws error."""
        mock_predictor.predict_batch.side_effect = Exception("Batch prediction failed")
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict_batch",
            json=batch_media_input
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Batch prediction error" in data["message"]


class TestDocumentationEndpoints:
    """Test documentation endpoints."""
    
    def test_openapi_json(self, client_with_mock_predictor):
        """Test OpenAPI JSON endpoint."""
        response = client_with_mock_predictor.get("/reel-driver/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Reel Driver API"
    
    def test_swagger_docs(self, client_with_mock_predictor):
        """Test Swagger UI docs endpoint."""
        response = client_with_mock_predictor.get("/reel-driver/docs")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self, client_with_mock_predictor):
        """Test ReDoc docs endpoint."""
        response = client_with_mock_predictor.get("/reel-driver/redoc")
        
        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_404_endpoint(self, client_with_mock_predictor):
        """Test non-existent endpoint returns 404."""
        response = client_with_mock_predictor.get("/reel-driver/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_invalid_method(self, client_with_mock_predictor):
        """Test invalid HTTP method on prediction endpoint."""
        response = client_with_mock_predictor.get("/reel-driver/api/predict")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
    
    def test_malformed_json(self, client_with_mock_predictor):
        """Test malformed JSON request."""
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            data="invalid json"
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestInputValidation:
    """Test input validation edge cases."""
    
    def test_country_code_validation(self, client_with_mock_predictor):
        """Test country code validation."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "origin_country": ["USA"]  # Should be 2 chars
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_language_code_validation(self, client_with_mock_predictor):
        """Test language code validation."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "original_language": "eng"  # Should be 2 chars
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_genre_length_validation(self, client_with_mock_predictor):
        """Test genre name length validation."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "genre": ["A very long genre name that exceeds the limit"]
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_negative_values(self, client_with_mock_predictor):
        """Test validation of negative values."""
        invalid_input = {
            "imdb_id": "tt0111161",
            "budget": -1000000  # Should be >= 0
        }
        
        response = client_with_mock_predictor.post(
            "/reel-driver/api/predict",
            json=invalid_input
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY