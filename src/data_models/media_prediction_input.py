# standard library imports
from typing import List, Optional

# third party imports
from pydantic import BaseModel, Field, field_validator


class MediaPredictionInput(BaseModel):
    """
    Input model for media prediction requests.
    Contains raw media metadata that will be transformed for model prediction.
    """
    hash: str = Field(
        description="Unique identifier for the media item"
    )

    release_year: Optional[int] = Field(
        default=None,
        description="Year the media was released"
    )

    genre: Optional[List[str]] = Field(
        default=None,
        description="List of genres for the media item"
    )

    language: Optional[List[str]] = Field(
        default=None,
        description="List of language codes in ISO 639 format"
    )

    metascore: Optional[int] = Field(
        default=None,
        description="Metascore rating (0-100)"
    )

    rt_score: Optional[int] = Field(
        default=None,
        description="Rotten Tomatoes score (0-100)"
    )

    imdb_rating: Optional[float] = Field(
        default=None,
        description="IMDB rating (0-10)"
    )

    imdb_votes: Optional[int] = Field(
        default=None,
        description="Number of votes on IMDB"
    )

    # Field validators
    @field_validator('metascore')
    @classmethod
    def check_metascore_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 100):
            raise ValueError('metascore must be between 0 and 100')
        return v

    @field_validator('rt_score')
    @classmethod
    def check_rt_score_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 0 or v > 100):
            raise ValueError('rt_score must be between 0 and 100')
        return v

    @field_validator('imdb_rating')
    @classmethod
    def check_imdb_rating_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 10):
            raise ValueError('imdb_rating must be between 0 and 10')
        return v

    @field_validator('imdb_votes')
    @classmethod
    def check_imdb_votes_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError('imdb_votes must be greater than or equal to 0')
        return v

    @field_validator('release_year')
    @classmethod
    def check_release_year_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1900 or v > 2100):
            raise ValueError('release_year must be between 1900 and 2100')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "hash": "e0bf77125af67cd5d3a33d44f58ee117d982df22",
                "release_year": 1980,
                "genre": ["Animation", "Fantasy", "Adventure"],
                "language": ["en"],
                "rt_score": 67,
                "metascore": None,
                "imdb_rating": 5.7,
                "imdb_votes": 4815
            }
        }