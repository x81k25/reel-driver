#!/usr/bin/env python3
"""
Test script to verify FastAPI startup, health check, and shutdown.
This script tests the actual API server startup process.
"""

import os
import sys
import time
import subprocess
import requests
import signal
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

def test_api_startup():
    """Test the FastAPI server startup, health check, and shutdown."""
    
    print("🚀 Starting FastAPI test...")
    
    # Configuration
    host = "127.0.0.1"
    port = 8001  # Use different port to avoid conflicts
    startup_timeout = 60  # seconds
    health_check_url = f"http://{host}:{port}/reel-driver/health"
    
    # Start the FastAPI server
    print(f"📡 Starting FastAPI server on {host}:{port}")
    
    env = os.environ.copy()
    env['REEL_DRIVER_API_HOST'] = host
    env['REEL_DRIVER_API_PORT'] = str(port)
    
    # Ensure PYTHONPATH includes project root
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
    else:
        env['PYTHONPATH'] = project_root
    
    try:
        # Start server process
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", 
             "--host", host, "--port", str(port), "--log-level", "info"],
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"📋 Server process started with PID: {process.pid}")
        
        # Wait for server to start up
        print("⏳ Waiting for server startup...")
        startup_success = False
        
        for attempt in range(startup_timeout):
            try:
                response = requests.get(health_check_url, timeout=5)
                if response.status_code == 200:
                    startup_success = True
                    print(f"✅ Server started successfully after {attempt + 1} seconds")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Server process died during startup")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
        
        if not startup_success:
            print(f"❌ Server failed to start within {startup_timeout} seconds")
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
        # Test health check endpoint
        print("🏥 Testing health check endpoint...")
        try:
            response = requests.get(health_check_url, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health check passed!")
                print(f"   Status: {health_data.get('status')}")
                print(f"   Model: {health_data.get('model_name')} v{health_data.get('model_version')}")
                print(f"   Features: {health_data.get('features_count')}")
                print(f"   Run ID: {health_data.get('run_id')}")
                
                # Verify health data structure
                required_fields = ['status', 'model_name', 'model_version', 'run_id', 'features_count']
                missing_fields = [field for field in required_fields if field not in health_data]
                
                if missing_fields:
                    print(f"⚠️  Missing health check fields: {missing_fields}")
                    return False
                
                if health_data['status'] != 'healthy':
                    print(f"⚠️  Health status is not 'healthy': {health_data['status']}")
                    return False
                
                print("✅ Health check data validation passed!")
                
            else:
                print(f"❌ Health check failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check request failed: {e}")
            return False
        
        # Test root endpoint
        print("🏠 Testing root endpoint...")
        try:
            root_url = f"http://{host}:{port}/reel-driver/"
            response = requests.get(root_url, timeout=5)
            
            if response.status_code == 200:
                root_data = response.json()
                print("✅ Root endpoint passed!")
                print(f"   Message: {root_data.get('message')}")
                print(f"   Version: {root_data.get('version')}")
            else:
                print(f"⚠️  Root endpoint returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Root endpoint request failed: {e}")
        
        print("🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
        
    finally:
        # Shutdown server
        print("🛑 Shutting down server...")
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print("✅ Server shut down gracefully")
                except subprocess.TimeoutExpired:
                    print("⚠️  Server didn't shut down gracefully, forcing termination")
                    process.kill()
                    process.wait()
                    
            else:
                print("ℹ️  Server process already terminated")
                
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")


def main():
    """Main function to run the API startup test."""
    print("=" * 60)
    print("FastAPI Startup Integration Test")
    print("=" * 60)
    
    # Check environment configuration
    required_vars = [
        'REEL_DRIVER_MLFLOW_HOST',
        'REEL_DRIVER_MLFLOW_PORT',
        'REEL_DRIVER_MLFLOW_EXPERIMENT',
        'REEL_DRIVER_MLFLOW_MODEL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file configuration")
        return False
    
    print("✅ Environment configuration validated")
    
    # Run the test
    success = test_api_startup()
    
    print("=" * 60)
    if success:
        print("🎉 API STARTUP TEST PASSED!")
        return True
    else:
        print("❌ API STARTUP TEST FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)