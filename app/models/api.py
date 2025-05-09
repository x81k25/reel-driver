from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.models.media_prediction_input import MediaPredictionInput

class PredictionResponse(BaseModel):
    """Response model for a single prediction."""
    hash: str
    would_watch: bool
    probability: float

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    items: List[MediaPredictionInput]

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[Dict[str, Any]]