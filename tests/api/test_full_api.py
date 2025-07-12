import pytest
import os
import sys
import time
import subprocess
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load environment variables
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


@pytest.mark.integration
class TestFullAPI:
    """Complete API test suite that starts the server and tests all endpoints sequentially."""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start FastAPI server once for all tests in this class."""
        print("\n🚀 Starting FastAPI server for full endpoint testing...")
        
        # Configuration
        host = "127.0.0.1"
        port = 8003  # Use unique port
        startup_timeout = 60  # seconds
        
        env = os.environ.copy()
        env['REEL_DRIVER_API_HOST'] = host
        env['REEL_DRIVER_API_PORT'] = str(port)
        
        # Ensure PYTHONPATH includes project root
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = project_root
        
        # Start server
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", 
             "--host", host, "--port", str(port), "--log-level", "warning"],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Store server info
        base_url = f"http://{host}:{port}"
        
        # Wait for startup
        health_url = f"{base_url}/reel-driver/health"
        startup_success = False
        
        print(f"⏳ Waiting for server startup on {base_url}...")
        for attempt in range(startup_timeout):
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    startup_success = True
                    print(f"✅ Server started successfully after {attempt + 1} seconds")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            
            # Check if process died
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Server process died during startup. STDOUT: {stdout}, STDERR: {stderr}")
        
        if not startup_success:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"Server failed to start within {startup_timeout}s. STDOUT: {stdout}, STDERR: {stderr}")
        
        yield base_url
        
        # Cleanup
        print("\n🛑 Shutting down FastAPI server...")
        process.terminate()
        try:
            process.wait(timeout=10)
            print("✅ Server shut down gracefully")
        except subprocess.TimeoutExpired:
            process.kill()
            print("⚠️ Server forced shutdown")

    def test_01_health_endpoint(self, api_server):
        """Test the health check endpoint."""
        print("\n🏥 Testing health endpoint...")
        
        response = requests.get(f"{api_server}/reel-driver/health", timeout=10)
        
        assert response.status_code == 200, f"Health check failed with status {response.status_code}"
        
        data = response.json()
        print(f"Response: {data}")
        
        # Verify required fields
        required_fields = ['status', 'model_name', 'model_version', 'run_id', 'features_count']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data['status'] == 'healthy', f"Expected healthy status, got {data['status']}"
        assert data['features_count'] > 0, "Features count should be greater than 0"
        assert data['model_name'] == 'reel_driver', f"Expected model_name 'reel_driver', got {data['model_name']}"
        
        print(f"✅ Health check passed: {data['model_name']} v{data['model_version']} with {data['features_count']} features")

    def test_02_root_endpoint(self, api_server):
        """Test the root endpoint."""
        print("\n🏠 Testing root endpoint...")
        
        response = requests.get(f"{api_server}/reel-driver/", timeout=5)
        
        assert response.status_code == 200, f"Root endpoint failed with status {response.status_code}"
        
        data = response.json()
        print(f"Response: {data}")
        
        assert 'message' in data, "Missing 'message' field"
        assert 'version' in data, "Missing 'version' field"
        assert 'endpoints' in data, "Missing 'endpoints' field"
        assert data['message'] == 'Welcome to Reel Driver API', f"Unexpected message: {data['message']}"
        
        print(f"✅ Root endpoint passed: {data['message']} v{data['version']}")

    def test_03_openapi_json_endpoint(self, api_server):
        """Test the OpenAPI JSON endpoint."""
        print("\n📄 Testing OpenAPI JSON endpoint...")
        
        response = requests.get(f"{api_server}/reel-driver/openapi.json", timeout=5)
        
        assert response.status_code == 200, f"OpenAPI endpoint failed with status {response.status_code}"
        
        data = response.json()
        
        assert 'openapi' in data, "Missing 'openapi' field"
        assert 'info' in data, "Missing 'info' field"
        assert data['info']['title'] == 'Reel Driver API', f"Unexpected title: {data['info']['title']}"
        
        print(f"✅ OpenAPI JSON passed: {data['info']['title']} v{data['info']['version']}")

    def test_04_swagger_docs_endpoint(self, api_server):
        """Test the Swagger UI docs endpoint."""
        print("\n📚 Testing Swagger docs endpoint...")
        
        response = requests.get(f"{api_server}/reel-driver/docs", timeout=5)
        
        assert response.status_code == 200, f"Swagger docs failed with status {response.status_code}"
        assert "text/html" in response.headers.get("content-type", ""), "Expected HTML content"
        assert "swagger" in response.text.lower(), "Expected Swagger UI content"
        
        print("✅ Swagger docs endpoint accessible")

    def test_05_redoc_docs_endpoint(self, api_server):
        """Test the ReDoc docs endpoint."""
        print("\n📖 Testing ReDoc docs endpoint...")
        
        response = requests.get(f"{api_server}/reel-driver/redoc", timeout=5)
        
        assert response.status_code == 200, f"ReDoc docs failed with status {response.status_code}"
        assert "text/html" in response.headers.get("content-type", ""), "Expected HTML content"
        assert "redoc" in response.text.lower(), "Expected ReDoc content"
        
        print("✅ ReDoc docs endpoint accessible")

    def test_06_single_prediction_minimal(self, api_server):
        """Test single prediction with minimal valid input."""
        print("\n🎯 Testing single prediction endpoint (minimal input)...")
        
        test_data = {
            "imdb_id": "tt0111161"
        }
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict", 
            json=test_data,
            timeout=15
        )
        
        assert response.status_code == 200, f"Prediction failed with status {response.status_code}: {response.text}"
        
        data = response.json()
        print(f"Response: {data}")
        
        # Verify response structure
        required_fields = ['imdb_id', 'prediction', 'probability']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data['imdb_id'] == "tt0111161", f"Expected imdb_id 'tt0111161', got {data['imdb_id']}"
        assert isinstance(data['prediction'], bool), f"Expected boolean prediction, got {type(data['prediction'])}"
        assert isinstance(data['probability'], float), f"Expected float probability, got {type(data['probability'])}"
        assert 0.0 <= data['probability'] <= 1.0, f"Probability {data['probability']} not in range [0, 1]"
        
        print(f"✅ Single prediction passed: {data['prediction']} (probability: {data['probability']:.3f})")

    def test_07_single_prediction_full(self, api_server):
        """Test single prediction with full input data."""
        print("\n🎯 Testing single prediction endpoint (full input)...")
        
        test_data = {
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
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict", 
            json=test_data,
            timeout=15
        )
        
        assert response.status_code == 200, f"Full prediction failed with status {response.status_code}: {response.text}"
        
        data = response.json()
        print(f"Response: {data}")
        
        assert data['imdb_id'] == "tt0111161"
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['probability'], float)
        assert 0.0 <= data['probability'] <= 1.0
        
        print(f"✅ Full prediction passed: {data['prediction']} (probability: {data['probability']:.3f})")

    def test_08_batch_prediction(self, api_server):
        """Test batch prediction endpoint."""
        print("\n🎯 Testing batch prediction endpoint...")
        
        test_data = {
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
                },
                {
                    "imdb_id": "tt0468569",
                    "release_year": 2008,
                    "genre": ["Action", "Crime", "Drama"],
                    "runtime": 152
                }
            ]
        }
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict_batch", 
            json=test_data,
            timeout=20
        )
        
        assert response.status_code == 200, f"Batch prediction failed with status {response.status_code}: {response.text}"
        
        data = response.json()
        print(f"Response: {data}")
        
        assert 'results' in data, "Missing 'results' field"
        assert len(data['results']) == 3, f"Expected 3 results, got {len(data['results'])}"
        
        for i, result in enumerate(data['results']):
            expected_imdb_id = test_data['items'][i]['imdb_id']
            
            assert 'imdb_id' in result, f"Missing 'imdb_id' in result {i}"
            assert 'prediction' in result, f"Missing 'prediction' in result {i}"
            assert 'probability' in result, f"Missing 'probability' in result {i}"
            
            assert result['imdb_id'] == expected_imdb_id, f"Expected {expected_imdb_id}, got {result['imdb_id']}"
            assert isinstance(result['prediction'], bool), f"Expected boolean prediction in result {i}"
            assert isinstance(result['probability'], float), f"Expected float probability in result {i}"
            assert 0.0 <= result['probability'] <= 1.0, f"Probability {result['probability']} not in range [0, 1] for result {i}"
            
            print(f"  Result {i+1}: {result['imdb_id']} -> {result['prediction']} ({result['probability']:.3f})")
        
        print(f"✅ Batch prediction passed with {len(data['results'])} results")

    def test_09_validation_errors(self, api_server):
        """Test input validation errors."""
        print("\n⚠️ Testing input validation...")
        
        # Test invalid IMDB ID format
        invalid_data = {"imdb_id": "invalid_id"}
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict", 
            json=invalid_data,
            timeout=10
        )
        
        assert response.status_code == 422, f"Expected validation error 422, got {response.status_code}"
        print("✅ Invalid IMDB ID correctly rejected")
        
        # Test invalid year
        invalid_year_data = {
            "imdb_id": "tt0111161",
            "release_year": 1800  # Below minimum
        }
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict", 
            json=invalid_year_data,
            timeout=10
        )
        
        assert response.status_code == 422, f"Expected validation error 422, got {response.status_code}"
        print("✅ Invalid year correctly rejected")

    def test_10_nonexistent_endpoints(self, api_server):
        """Test that non-existent endpoints return 404."""
        print("\n❌ Testing non-existent endpoints...")
        
        response = requests.get(f"{api_server}/reel-driver/nonexistent", timeout=5)
        assert response.status_code == 404, f"Expected 404 for non-existent endpoint, got {response.status_code}"
        
        print("✅ Non-existent endpoints correctly return 404")

    def test_11_method_not_allowed(self, api_server):
        """Test invalid HTTP methods."""
        print("\n❌ Testing invalid HTTP methods...")
        
        # GET on prediction endpoint (should be POST)
        response = requests.get(f"{api_server}/reel-driver/api/predict", timeout=5)
        assert response.status_code == 405, f"Expected 405 for invalid method, got {response.status_code}"
        
        print("✅ Invalid HTTP methods correctly return 405")

    def test_12_empty_batch(self, api_server):
        """Test batch prediction with empty items list."""
        print("\n📝 Testing empty batch prediction...")
        
        empty_batch = {"items": []}
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict_batch", 
            json=empty_batch,
            timeout=10
        )
        
        assert response.status_code == 200, f"Empty batch failed with status {response.status_code}"
        
        data = response.json()
        assert 'results' in data
        assert data['results'] == [], f"Expected empty results, got {data['results']}"
        
        print("✅ Empty batch prediction handled correctly")

    def test_13_malformed_json(self, api_server):
        """Test malformed JSON request."""
        print("\n❌ Testing malformed JSON...")
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        assert response.status_code == 422, f"Expected 422 for malformed JSON, got {response.status_code}"
        print("✅ Malformed JSON correctly rejected")