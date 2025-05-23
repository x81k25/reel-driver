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
from app.services.predictor import XGBMediaPredictor
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
    # Load model first
    predictor = XGBMediaPredictor(artifacts_path=settings.MODEL_PATH)
    logger.info("Model loaded successfully")

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
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Additional health checks for model artifacts
    try:
        if not hasattr(predictor, 'feature_names') or not predictor.feature_names:
            raise HTTPException(status_code=503, detail="Model features not properly loaded")
        if not hasattr(predictor, 'normalization') or not predictor.normalization:
            raise HTTPException(status_code=503, detail="Normalization parameters not loaded")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Model health check failed")

    return {
        "status": "healthy",
        "model_features": len(predictor.feature_names),
        "normalization_fields": len(predictor.normalization)
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

# Import routers at the end to avoid circular imports
from app.routers import prediction

# Include routers - only once
app.include_router(prediction.get_router(predictor), prefix="/reel-driver/api", tags=["prediction"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True
    )