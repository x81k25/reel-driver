import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from app.models.media_prediction_input import MediaPredictionInput
from app.models.api import PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.services.predictor import XGBMediaPredictor

# Use module-level logger instead of importing from main
logger = logging.getLogger("uvicorn.error")


def get_router(predictor: XGBMediaPredictor):
    router = APIRouter()

    @router.post("/predict")
    async def predict(media_input: MediaPredictionInput):
        logger.info("prediction request received")
        
        # Use current global predictor instead of captured predictor for test isolation
        from app.main import predictor as current_predictor
        import app.main as main_module

        if current_predictor is None:
            # Attempt to reinitialize predictor if it's None
            logger.info("Predictor is None, attempting to reinitialize...")
            try:
                from app.services.predictor import XGBMediaPredictor
                new_predictor = XGBMediaPredictor()
                main_module.predictor = new_predictor
                current_predictor = new_predictor
                logger.info(f"Predictor successfully reinitialized with model v{new_predictor.loaded_model_version}")
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize predictor: {reinit_error}")
                raise HTTPException(
                    status_code=503, 
                    detail="Model not loaded - MLflow connection failed or model not found. Check MLflow server status and model registry."
                )

        try:
            # Check for model updates before prediction
            current_predictor.ensure_latest_model()
            
            result = current_predictor.predict(media_input)
            return PredictionResponse(
                imdb_id=result["imdb_id"],
                prediction=result["prediction"],
                probability=result["probability"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @router.post("/predict_batch", response_model=BatchPredictionResponse, status_code=200)
    async def predict_batch(request: BatchPredictionRequest):
        """
        Predict whether user would watch each media item in a batch.
        """
        logger.info("Batch prediction request received")
        
        # Use current global predictor instead of captured predictor for test isolation
        from app.main import predictor as current_predictor
        import app.main as main_module
        logger.info(f"Predictor is None: {current_predictor is None}")

        if current_predictor is None:
            # Attempt to reinitialize predictor if it's None
            logger.info("Predictor is None in batch prediction, attempting to reinitialize...")
            try:
                from app.services.predictor import XGBMediaPredictor
                new_predictor = XGBMediaPredictor()
                main_module.predictor = new_predictor
                current_predictor = new_predictor
                logger.info(f"Predictor successfully reinitialized with model v{new_predictor.loaded_model_version}")
            except Exception as reinit_error:
                logger.error(f"Failed to reinitialize predictor in batch prediction: {reinit_error}")
                raise HTTPException(
                    status_code=503, 
                    detail="Model not loaded - MLflow connection failed or model not found. Check MLflow server status and model registry."
                )

        try:
            # Check for model updates before batch prediction
            current_predictor.ensure_latest_model()
            
            results = current_predictor.predict_batch(request.items)
            return BatchPredictionResponse(results=results)
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    return router