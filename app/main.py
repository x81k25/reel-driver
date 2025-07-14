# standard library imports
import logging
import os

# third party imports
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import uvicorn.logging

# custom/local imports
from app.services.predictor import XGBMediaPredictor, ModelLoadingError
from app.services.mlflow_client import (
    MLflowConnectionError, 
    MLflowModelNotFoundError, 
    MLflowArtifactError
)
from app.core.config import settings

# Load environment variables
load_dotenv()

# Configure uvicorn access logs
logger = logging.getLogger("uvicorn.error")

# Initialize predictor as a global variable
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    
    # Check if we're in test mode by looking for a test marker
    test_mode = getattr(app.state, 'test_mode', False)
    
    if not test_mode and predictor is None:
        # Load model from MLflow in production mode
        try:
            predictor = XGBMediaPredictor()
            logger.info(f"Model {predictor.model_name} v{predictor.loaded_model_version} loaded successfully from MLflow")
        except MLflowConnectionError as e:
            logger.error(f"MLflow connection failed during startup: {e}")
            # Don't raise - let the app start but health checks will fail
            predictor = None
        except MLflowModelNotFoundError as e:
            logger.error(f"MLflow model not found during startup: {e}")
            # Don't raise - let the app start but health checks will fail
            predictor = None
        except MLflowArtifactError as e:
            logger.error(f"MLflow artifacts failed during startup: {e}")
            # Don't raise - let the app start but health checks will fail
            predictor = None
        except ModelLoadingError as e:
            logger.error(f"Model loading failed during startup: {e}")
            # Don't raise - let the app start but health checks will fail
            predictor = None
        except Exception as e:
            logger.error(f"Unexpected error during model loading: {e}")
            # Don't raise - let the app start but health checks will fail
            predictor = None
    else:
        if test_mode:
            logger.info("Running in test mode - using test predictor configuration")
        else:
            logger.info("Using existing predictor")

    # Include router AFTER predictor is loaded
    from app.routers import prediction
    app.include_router(prediction.get_router(predictor), prefix="/reel-driver/api", tags=["prediction"])

    yield
    logger.info("Shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Reel Driver API",
    description="Personal media curation API for predicting media preferences using IMDB metadata",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Create a root router with the prefix
root_router = APIRouter(prefix="/reel-driver")

# Then, after including all the routers, add this:
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

@root_router.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )

# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

@app.exception_handler(MLflowConnectionError)
async def mlflow_connection_exception_handler(request, exc):
    logger.error(f"MLflow connection error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "message": "MLflow service unavailable - cannot connect to MLflow server",
            "details": str(exc),
            "service": "mlflow"
        },
    )

@app.exception_handler(MLflowModelNotFoundError)
async def mlflow_model_not_found_exception_handler(request, exc):
    logger.error(f"MLflow model not found: {exc}")
    return JSONResponse(
        status_code=404,
        content={
            "message": "Model not found in MLflow registry",
            "details": str(exc),
            "service": "mlflow"
        },
    )

@app.exception_handler(MLflowArtifactError)
async def mlflow_artifact_exception_handler(request, exc):
    logger.error(f"MLflow artifact error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "message": "MLflow model artifacts unavailable - cannot download required model files",
            "details": str(exc),
            "service": "mlflow"
        },
    )

@app.exception_handler(ModelLoadingError)
async def model_loading_exception_handler(request, exc):
    logger.error(f"Model loading error: {exc}")
    return JSONResponse(
        status_code=503,
        content={
            "message": "Model loading failed - check MLflow connectivity and model artifacts",
            "details": str(exc),
            "service": "model"
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

# Health check endpoint
@root_router.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded - MLflow connection failed or model not found. Check MLflow server status and model registry."
        )

    # Additional health checks for MLflow model artifacts
    try:
        if not hasattr(predictor, 'feature_names') or not predictor.feature_names:
            raise HTTPException(
                status_code=503, 
                detail="Model features not properly loaded - check MLflow model artifacts"
            )
        if not hasattr(predictor, 'normalization') or not predictor.normalization:
            raise HTTPException(
                status_code=503, 
                detail="Normalization parameters not loaded - check MLflow model artifacts"
            )
        if not hasattr(predictor, 'engineered_schema') or not predictor.engineered_schema:
            raise HTTPException(
                status_code=503, 
                detail="Engineered schema not loaded - check MLflow model artifacts"
            )
        if not hasattr(predictor, 'model') or predictor.model is None:
            raise HTTPException(
                status_code=503, 
                detail="XGBoost model not loaded - check MLflow model registry"
            )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail="Model health check failed - check MLflow connectivity and model artifacts"
        )

    return {
        "status": "healthy",
        "model_name": predictor.model_name,
        "model_version": predictor.loaded_model_version,
        "run_id": predictor.run_id,
        "features_count": len(predictor.feature_names),
        "normalization_fields": len(predictor.normalization),
        "schema_mappings": len(predictor.engineered_schema),
        "genre_categories": len(predictor.genres),
        "origin_countries": len(predictor.origin_countries),
        "production_countries": len(predictor.production_countries),
        "spoken_languages": len(predictor.spoken_languages)
    }

# Root endpoint
@root_router.get("/", tags=["root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Welcome to Reel Driver API",
        "description": "Personal media curation API for predicting media preferences",
        "version": "0.1.0",
        "docs": "/reel-driver/docs",
        "health": "/reel-driver/health",
        "endpoints": {
            "single_prediction": "/reel-driver/api/predict",
            "batch_prediction": "/reel-driver/api/predict_batch"
        }
    }

# Mount docs at the prefixed path
@root_router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/reel-driver/openapi.json",
        title=app.title + " - Swagger UI"
    )

@root_router.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    return get_redoc_html(
        openapi_url="/reel-driver/openapi.json",
        title=app.title + " - ReDoc"
    )

# Include the root router in the app
app.include_router(root_router)

# REMOVED: Duplicate router inclusion that was causing the issue
# The prediction router is now only included in the lifespan function

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.REEL_DRIVER_API_HOST,
        port=settings.REEL_DRIVER_API_PORT,
        reload=True
    )