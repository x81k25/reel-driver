# reel-driver

FastAPI inference service for the Reel Driver media recommendation system.

## Overview

This project provides a production-ready REST API for serving predictions from trained XGBoost models. It loads models from MLflow and serves real-time predictions for media content recommendations.

For ML training components (feature engineering, model training, notebooks), see the [reel-driver-modeling](https://github.com/your-org/reel-driver-modeling) repository.

## Project Structure

```
reel-driver/
├── app/                               # FastAPI inference service
│   ├── core/                         # Configuration and settings
│   ├── models/                       # Pydantic data models
│   ├── routers/                      # API route handlers
│   ├── services/                     # Business logic and ML services
│   └── main.py                       # FastAPI application entry point
├── tests/                            # Test suite
│   ├── api/                         # API endpoint tests
│   │   ├── test_endpoints.py        # Unit tests with mocked predictors
│   │   ├── test_full_api.py         # Full integration tests
│   │   └── test_integration.py      # MLflow integration tests
│   └── conftest.py                  # Shared test fixtures
├── containerization/                # Docker container definitions
│   └── dockerfile.api               # API service image
├── docs/                            # Documentation
├── pyproject.toml                   # Project configuration and dependencies
├── uv.lock                         # Locked dependencies for reproducible builds
└── CLAUDE.md                       # Detailed technical documentation
```

## Quick Start

### Prerequisites

- Python 3.12+
- MLflow server (for model loading)
- MinIO or S3-compatible storage (for MLflow artifacts)

### Environment Variables

Create a `.env` file in the root directory:

```bash
# MLflow Configuration
REEL_DRIVER_MLFLOW_HOST=mlflow-host
REEL_DRIVER_MLFLOW_PORT=5000
REEL_DRIVER_MLFLOW_EXPERIMENT=reel-driver-experiment
REEL_DRIVER_MLFLOW_MODEL=reel-driver-model

# MinIO Configuration (for MLflow artifacts)
REEL_DRIVER_MINIO_ENDPOINT=http://minio-host
REEL_DRIVER_MINIO_PORT=9000
REEL_DRIVER_MINIO_ACCESS_KEY=your-access-key
REEL_DRIVER_MINIO_SECRET_KEY=your-secret-key

# Development Mode
LOCAL_DEVELOPMENT=true  # Set to 'true' for local development
```

### Installation

This project uses `uv` for dependency management:

```bash
# Create and install dependencies
uv sync

# Note: uv automatically creates and manages the virtual environment in .venv
# To activate the environment manually: source .venv/bin/activate
```

## Testing

The project includes a comprehensive test suite:

### Run All Tests
```bash
# Run complete test suite
uv run pytest tests/api/

# Run with verbose output
uv run pytest tests/api/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only (fast, mocked dependencies)
uv run pytest tests/api/test_endpoints.py -v

# Full API tests
uv run pytest tests/api/test_full_api.py -v
```

### Integration Tests
Integration tests require real services. Set `LOCAL_DEVELOPMENT=true` in your `.env`:

```bash
# MLflow integration tests
uv run pytest tests/api/test_integration.py -v
```

## API Usage

### Start the API Server
```bash
# Local development
uv run uvicorn app.main:app --reload --port 8000

# The API will be available at http://localhost:8000/reel-driver
```

### API Endpoints

- **Health Check**: `GET /reel-driver/health`
- **Model Info**: `GET /reel-driver/api/model`
- **Single Prediction**: `POST /reel-driver/api/predict`
- **Batch Prediction**: `POST /reel-driver/api/predict_batch`

See [CLAUDE.md](./CLAUDE.md) for detailed API documentation.

## Container Deployment

### Build Image
```bash
docker build -f containerization/dockerfile.api -t reel-driver-api .
```

### Run Container
```bash
docker run --env-file .env -p 8000:8000 reel-driver-api
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive technical documentation
- **[tests/README.md](./tests/README.md)** - Testing documentation

## Architecture

This project implements the inference layer of an MLOps pipeline:

1. **FastAPI Inference Service** - Production-ready REST API
2. **MLflow Integration** - Model loading from registry
3. **Comprehensive Testing** - Unit tests, integration tests, and CI/CD
4. **Container Deployment** - Docker-based deployment for Kubernetes
