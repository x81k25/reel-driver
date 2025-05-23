from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from app.main import logger
from app.models.media_prediction_input import MediaPredictionInput
from app.models.api import PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.services.predictor import XGBMediaPredictor


def get_router(predictor):
    router = APIRouter()

    @router.post("/predict")
    async def predict(media_input: MediaPredictionInput):
        # Import here to avoid circular dependency
        logger.info("prediction request received")

        from app.main import predictor as main_predictor

        # Use either the passed predictor or the global one
        pred = predictor if predictor is not None else main_predictor

        if pred is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            result = pred.predict(media_input)
            return PredictionResponse(
                imdb_id=result["imdb_id"],
                prediction=result["prediction"],
                probability=result["probability"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


    # Other route handlers with similar imports
    @router.post("/predict_batch", response_model=BatchPredictionResponse, status_code=200)
    async def predict_batch(request: BatchPredictionRequest):
        """
        Predict whether user would watch each media item in a batch.
        """
        logger.info("Batch prediction request received")

        from app.main import predictor as main_predictor

        # Use either the passed predictor or the global one
        pred = predictor if predictor is not None else main_predictor

        logger.info(f"Predictor is None: {pred is None}")

        if pred is None:
            logger.error("Model not loaded in batch prediction")
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            results = pred.predict_batch(request.items)
            return BatchPredictionResponse(results=results)
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    return router