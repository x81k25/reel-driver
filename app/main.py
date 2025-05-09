# standard library imports
import logging
import os

# third party imports
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn.logging

# custom/local imports
from app.services.predictor import XGBMediaPredictor
from app.core.config import settings

# Load environment variables
load_dotenv()

# Configure uvicorn access logs
uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.setLevel(logging.INFO)

# Create a custom logger with a unique name
logger = logging.getLogger("reel_driver")
# Ensure your messages appear
logger.setLevel(logging.INFO)
# Add handler if it doesn't have one
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s:     %(message)s'))
    logger.addHandler(handler)

# Initialize predictor as a global variable
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global predictor
    # Startup logic
    logger.info("Loading XGBoost model...")
    try:
        predictor = XGBMediaPredictor(artifacts_path=settings.MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Could not load model: {e}")

    yield  # This is where the app runs

    # Shutdown logic
    logger.info("Shutting down API")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Reel Driver API",
    description="Personal media curation API for predicting media preferences",
    version="0.1.0",
    lifespan=lifespan
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
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
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    logger.info("receiving request for health check")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Welcome to Reel Driver API",
        "docs": "/docs",
        "health": "/health"
    }

# Import routers at the end to avoid circular imports
from app.routers import prediction

# Include routers - only once
app.include_router(prediction.get_router(predictor), prefix="/api", tags=["prediction"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True
    )