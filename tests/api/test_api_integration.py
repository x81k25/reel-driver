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
class TestAPIStartup:
    """Integration tests for FastAPI server startup and basic functionality."""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start and stop FastAPI server for testing."""
        # Configuration
        host = "127.0.0.1"
        port = 8002  # Different port to avoid conflicts
        
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
        
        # Wait for startup
        base_url = f"http://{host}:{port}"
        health_url = f"{base_url}/reel-driver/health"
        
        startup_success = False
        for _ in range(30):  # 30 second timeout
            try:
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    startup_success = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            
            if process.poll() is not None:
                break
        
        if not startup_success:
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"Server failed to start. STDOUT: {stdout}, STDERR: {stderr}")
        
        yield base_url
        
        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    
    def test_server_startup(self, api_server):
        """Test that the server starts successfully."""
        assert api_server is not None
        
    def test_health_endpoint(self, api_server):
        """Test the health check endpoint."""
        response = requests.get(f"{api_server}/reel-driver/health", timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = ['status', 'model_name', 'model_version', 'run_id', 'features_count']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data['status'] == 'healthy'
        assert data['features_count'] > 0
        assert data['model_name'] == 'reel_driver'
        
        print(f"Health check passed: {data['model_name']} v{data['model_version']}")
    
    def test_root_endpoint(self, api_server):
        """Test the root endpoint."""
        response = requests.get(f"{api_server}/reel-driver/", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'message' in data
        assert 'version' in data
        assert 'endpoints' in data
        assert data['message'] == 'Welcome to Reel Driver API'
        
        print(f"Root endpoint passed: {data['message']} v{data['version']}")
    
    def test_openapi_docs(self, api_server):
        """Test that OpenAPI docs are accessible."""
        response = requests.get(f"{api_server}/reel-driver/openapi.json", timeout=5)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'openapi' in data
        assert 'info' in data
        assert data['info']['title'] == 'Reel Driver API'
        
        print("OpenAPI documentation accessible")
    
    def test_prediction_endpoint_structure(self, api_server):
        """Test prediction endpoint with minimal valid data."""
        # Test with minimal valid input
        test_data = {
            "imdb_id": "tt0111161"
        }
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict", 
            json=test_data,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        required_fields = ['imdb_id', 'prediction', 'probability']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data['imdb_id'] == "tt0111161"
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['probability'], float)
        assert 0.0 <= data['probability'] <= 1.0
        
        print(f"Prediction endpoint works: {data['prediction']} (probability: {data['probability']:.3f})")
    
    def test_batch_prediction_endpoint_structure(self, api_server):
        """Test batch prediction endpoint with minimal valid data."""
        test_data = {
            "items": [
                {"imdb_id": "tt0111161"},
                {"imdb_id": "tt0068646"}
            ]
        }
        
        response = requests.post(
            f"{api_server}/reel-driver/api/predict_batch", 
            json=test_data,
            timeout=15
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'results' in data
        assert len(data['results']) == 2
        
        for i, result in enumerate(data['results']):
            assert 'imdb_id' in result
            assert 'prediction' in result
            assert 'probability' in result
            assert result['imdb_id'] == test_data['items'][i]['imdb_id']
            assert isinstance(result['prediction'], bool)
            assert isinstance(result['probability'], float)
            assert 0.0 <= result['probability'] <= 1.0
        
        print(f"Batch prediction endpoint works with {len(data['results'])} results")