# reel-driver

A personal media curation algorithm trained on personally labeled data.

## Overview

Ever opened app after app on your SmartTV and you were greeted by a top row of squares of content you mostly weren't interested in? Well, if so, then this may be the repo you've been looking for. I've been repeatedly disappointed by content curation on the big streamers. Either they promote content I'm not interested in, I have to dig for content I do want, or it may display content to me, which only to discover after clicking on it, that I need another subscription or a purchase to access it.

The intention of this project is to create a model that ingests personalized training data in order to create a model that can run inferences on new media items and tell you whether or not you'd be into it! The data samples I have in the `/data` folder contain the training data and analysis results based off of my own preferences; but you could easily recreate the results but altering the label field for your preferences or feeding in your own data.   

## Project Structure

The project is containerized for production deployment with training and inference components:

1. **Training Pipeline** - Feature engineering and model training containers
2. **API Service** - FastAPI inference service for model predictions  
3. **Testing Suite** - Comprehensive test coverage (66+ tests)

```
reel-driver/
├── src/                               # Training pipeline source code
│   ├── training/
│   │   ├── feature_engineering.py    # Feature engineering pipeline
│   │   └── model_training.py         # ML model training & tuning
│   └── utils/
│       └── db_operations.py          # Database utilities
├── app/                               # FastAPI inference service
│   ├── models/                       # Pydantic data models
│   ├── routers/                      # API route handlers
│   ├── services/                     # Business logic and ML services
│   └── main.py                       # FastAPI application entry point
├── tests/                            # Comprehensive test suite (66+ tests)
│   ├── api/                         # API endpoint tests
│   │   ├── test_endpoints.py        # Unit tests with mocked predictors
│   │   ├── test_full_api.py         # Full integration tests
│   │   └── test_integration.py      # MLflow integration tests
│   ├── training/                    # Training pipeline tests
│   │   ├── test_feature_engineering_integration.py
│   │   └── test_model_training_integration.py
│   └── conftest.py                  # Shared test fixtures
├── containerization/                # Docker container definitions
│   ├── dockerfile.base_training     # CPU base training image
│   ├── dockerfile.base_training_gpu # GPU base training image (CUDA 12.4)
│   ├── dockerfile.feature_engineering_cpu
│   ├── dockerfile.feature_engineering_gpu
│   ├── dockerfile.model_training_cpu
│   ├── dockerfile.model_training_gpu
│   └── dockerfile.api               # API service image
├── data/                           # Training data artifacts
├── notebooks/                      # Jupyter notebooks for analysis
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                        # Locked dependencies for reproducible builds
└── CLAUDE.md                      # Detailed technical documentation
```

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL database (for training pipeline)
- MLflow server (for model tracking and artifacts)
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

# Database Configuration (for training)
REEL_DRIVER_TRNG_PGSQL_HOST=postgresql-host
REEL_DRIVER_TRNG_PGSQL_PORT=5432
REEL_DRIVER_TRNG_PGSQL_DATABASE=database-name
REEL_DRIVER_TRNG_PGSQL_SCHEMA=schema-name
REEL_DRIVER_TRNG_PGSQL_USERNAME=username
REEL_DRIVER_TRNG_PGSQL_PASSWORD=password

# Development Mode
LOCAL_DEVELOPMENT=true  # Set to 'true' for local development
```

### Installation

This project uses `uv` for dependency management with modular dependencies:

```bash
# Create a virtual environment (optional - uv sync will create it automatically)
uv venv

# Install dependencies based on your needs:

# For API development/deployment
uv sync --extra api

# For feature engineering
uv sync --extra feature-engineering

# For model training
uv sync --extra model-training

# For running tests (includes dev dependencies)
uv sync

# For development with all extras
uv sync --all-extras

# Note: uv automatically creates and manages the virtual environment in .venv
# To activate the environment manually: source .venv/bin/activate
```

## Testing

The project includes a comprehensive test suite with 66+ tests:

### Run All Tests
```bash
# Run complete test suite
pytest

# Run with verbose output
pytest -v
```

### Run Specific Test Categories
```bash
# API tests only (unit + integration)
pytest tests/api/ -v

# Training pipeline tests only
pytest tests/training/ -v

# Unit tests only (fast, mocked dependencies)
pytest tests/api/test_endpoints.py -v
```

### Integration Tests
Integration tests require real services. Set `LOCAL_DEVELOPMENT=true` in your `.env`:

```bash
# MLflow integration tests
pytest tests/api/test_integration.py -v

# Training pipeline integration tests  
pytest tests/training/ -v
```

## API Usage

### Start the API Server
```bash
# Local development
uvicorn app.main:app --reload --port 8000

# The API will be available at http://localhost:8000/reel-driver
```

### API Endpoints

- **Health Check**: `GET /reel-driver/health`
- **Model Info**: `GET /reel-driver/api/model`
- **Single Prediction**: `POST /reel-driver/api/predict`
- **Batch Prediction**: `POST /reel-driver/api/predict_batch`

See [CLAUDE.md](./CLAUDE.md) for detailed API documentation.

## Training Pipeline

### Feature Engineering
```bash
python -c "from src.training.feature_engineering import __main__; __main__()"
```

### Model Training
```bash
python -c "from src.training.model_training import __main__; __main__()"
```

## Container Deployment

### Build Images
```bash
# Build CPU training containers
docker build -f containerization/dockerfile.base_training -t reel-driver-training-base-cpu .
docker build -f containerization/dockerfile.feature_engineering_cpu -t reel-driver-feature-engineering-cpu .
docker build -f containerization/dockerfile.model_training_cpu -t reel-driver-model-training-cpu .

# Build GPU training containers (requires NVIDIA Docker runtime)
docker build -f containerization/dockerfile.base_training_gpu -t reel-driver-training-base-gpu .
docker build -f containerization/dockerfile.feature_engineering_gpu -t reel-driver-feature-engineering-gpu .
docker build -f containerization/dockerfile.model_training_gpu -t reel-driver-model-training-gpu .

# Build API container
docker build -f containerization/dockerfile.api -t reel-driver-api .
```

### Run Containers
```bash
# Run CPU training containers
docker run --env-file .env reel-driver-feature-engineering-cpu
docker run --env-file .env reel-driver-model-training-cpu

# Run GPU training containers (requires nvidia-docker)
docker run --gpus all --env-file .env reel-driver-feature-engineering-gpu
docker run --gpus all --env-file .env reel-driver-model-training-gpu

# Run API container
docker run --env-file .env -p 8000:8000 reel-driver-api

# Or use docker-compose for local GPU/CPU training
docker-compose -f docker-compose.training.yaml --profile gpu up model-training-gpu
docker-compose -f docker-compose.training.yaml --profile cpu up model-training-cpu
```

## Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Comprehensive technical documentation
- **[tests/README.md](./tests/README.md)** - Testing documentation
- **Jupyter Notebooks** - In the `notebooks/` directory for data analysis

## Architecture

This project implements a modern MLOps pipeline:

1. **Stateless Training Containers** - Feature engineering and model training with CPU/GPU variants
2. **GPU Acceleration** - XGBoost training with CUDA support (22-25% speedup on current dataset)
3. **FastAPI Inference Service** - Production-ready REST API
4. **MLflow Integration** - Experiment tracking and model registry
5. **Comprehensive Testing** - Unit tests, integration tests, and CI/CD
6. **Container Orchestration** - Designed for Dagster pipeline orchestration

### Docker Images

| Image | Description |
|-------|-------------|
| `reel-driver-training-base-cpu` | CPU base image (Python 3.12) |
| `reel-driver-training-base-gpu` | GPU base image (CUDA 12.4) |
| `reel-driver-feature-engineering-cpu/gpu` | Feature engineering pipeline |
| `reel-driver-model-training-cpu/gpu` | XGBoost model training with Optuna |
| `reel-driver-api` | FastAPI inference service |

See [docs/CPU-v-GPU.md](./docs/CPU-v-GPU.md) for detailed GPU implementation documentation.