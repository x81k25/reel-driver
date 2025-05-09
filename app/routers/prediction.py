from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from app.models.media_prediction_input import MediaPredictionInput
from app.models.api import PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from app.services.predictor import XGBMediaPredictor

def get_router(predictor):
    router = APIRouter()

    @router.post("/predict", response_model=PredictionResponse, status_code=200)
    async def predict(
        media_input: MediaPredictionInput
    ):
        """
        Predict whether user would watch media based on provided metadata.
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            result = predictor.predict(media_input)
            return PredictionResponse(
                hash=result["hash"],
                would_watch=result["would_watch"],
                probability=result["probability"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    @router.post("/predict_batch", response_model=BatchPredictionResponse, status_code=200)
    async def predict_batch(
        request: BatchPredictionRequest
    ):
        """
        Predict whether user would watch each media item in a batch.
        """
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            results = predictor.predict_batch(request.items)
            return BatchPredictionResponse(results=results)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

    return router