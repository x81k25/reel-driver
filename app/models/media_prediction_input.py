# standard library imports
from typing import List, Optional
from decimal import Decimal

# third party imports
from pydantic import BaseModel, Field, field_validator


class MediaPredictionInput(BaseModel):
    """
    Input model for media prediction requests.
    Contains raw media metadata that will be transformed for model prediction.
    """
    # primary key
    imdb_id: str = Field(
        description="IMDB identifier for the media item",
        pattern=r'^tt[0-9]{7,8}$'
    )

    # time fields
    release_year: Optional[int] = Field(
        default=None,
        description="Year the media item was originally released",
        ge=1900,
        le=2100
    )

    # Quantitative details
    budget: Optional[int] = Field(
        default=None,
        description="Budget of the media item",
        ge=0
    )

    revenue: Optional[int] = Field(
        default=None,
        description="Revenue of the media item",
        ge=0
    )

    runtime: Optional[int] = Field(
        default=None,
        description="Runtime in minutes",
        ge=0
    )

    # Country and production information
    origin_country: Optional[List[str]] = Field(
        default=None,
        description="List of origin country codes (2-character ISO codes)"
    )

    production_companies: Optional[List[str]] = Field(
        default=None,
        description="List of production companies"
    )

    production_countries: Optional[List[str]] = Field(
        default=None,
        description="List of production country codes (2-character ISO codes)"
    )

    production_status: Optional[str] = Field(
        default=None,
        description="Production status of the media",
        max_length=25
    )

    # Language information
    original_language: Optional[str] = Field(
        default=None,
        description="Original language code (2-character ISO code)",
        max_length=2
    )

    spoken_languages: Optional[List[str]] = Field(
        default=None,
        description="List of spoken language codes (2-character ISO codes)"
    )

    # Other string fields
    genre: Optional[List[str]] = Field(
        default=None,
        description="List of genres for the media item"
    )

    # Long string fields
    tagline: Optional[str] = Field(
        default=None,
        description="Tagline of the media",
        max_length=255
    )

    overview: Optional[str] = Field(
        default=None,
        description="Overview/description of the media"
    )

    # Ratings information
    tmdb_rating: Optional[Decimal] = Field(
        default=None,
        description="TMDB rating (0-10)",
        ge=0,
        le=10,
        decimal_places=3
    )

    tmdb_votes: Optional[int] = Field(
        default=None,
        description="Number of votes on TMDB",
        ge=0
    )

    rt_score: Optional[int] = Field(
        default=None,
        description="Rotten Tomatoes score (0-100)",
        ge=0,
        le=100
    )

    metascore: Optional[int] = Field(
        default=None,
        description="Metascore rating (0-100)",
        ge=0,
        le=100
    )

    imdb_rating: Optional[Decimal] = Field(
        default=None,
        description="IMDB rating (0-100)",
        ge=0,
        le=100,
        decimal_places=1
    )

    imdb_votes: Optional[int] = Field(
        default=None,
        description="Number of votes on IMDB",
        ge=0
    )

    # Field validators for country codes
    @field_validator('origin_country', 'production_countries')
    @classmethod
    def validate_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for code in v:
                if len(code) != 2:
                    raise ValueError('Country codes must be 2 characters long')
        return v

    @field_validator('original_language')
    @classmethod
    def validate_language_code(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) != 2:
            raise ValueError('Language code must be 2 characters long')
        return v

    @field_validator('spoken_languages')
    @classmethod
    def validate_spoken_language_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for code in v:
                if len(code) != 2:
                    raise ValueError('Language codes must be 2 characters long')
        return v

    @field_validator('genre')
    @classmethod
    def validate_genres(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for genre in v:
                if len(genre) > 20:
                    raise ValueError('Genre names must be 20 characters or less')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "imdb_id": "tt0111161",
                "release_year": 2025,
                "budget": 25000000,
                "revenue": 16000000,
                "runtime": 142,
                "origin_country": ["US"],
                "production_companies": ["Castle Rock Entertainment"],
                "production_countries": ["US"],
                "production_status": "Released",
                "original_language": "en",
                "spoken_languages": ["en"],
                "genre": ["Drama", "Crime"],
                "tagline": "Fear can hold you prisoner. Hope can set you free.",
                "overview": "Framed in the 1940s for the double murder of his wife and her lover...",
                "tmdb_rating": 8.7,
                "tmdb_votes": 26000,
                "rt_score": 91,
                "metascore": 82,
                "imdb_rating": 92.0,
                "imdb_votes": 2800000
            }
        }