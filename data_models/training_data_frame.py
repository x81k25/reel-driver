from datetime import datetime, timezone
from enum import Enum
import polars as pl
from typing import List, Dict, Any, Optional, ClassVar, Type

# ------------------------------------------------------------------------------
# enum classes
# ------------------------------------------------------------------------------

pl.enable_string_cache()

class MediaType(str, Enum):
    MOVIE = 'movie'
    TV_SHOW = 'tv_show'
    TV_SEASON = 'tv_season'

# ------------------------------------------------------------------------------
# training dataframe class
# ------------------------------------------------------------------------------

class TrainingDataFrame:
    """Unified rigid polars DataFrame for training data matching SQL schemas."""

    # Complete schema with all possible fields
    schema = {
        # identifier columns
        'imdb_id': pl.Utf8,
        'tmdb_id': pl.Int64,
        # label columns
        'label': pl.Categorical,
        # media identifying information
        'media_type': pl.Categorical,
        'media_title': pl.Utf8,
        'season': pl.Int16,
        'episode': pl.Int16,
        'release_year': pl.Int16,
        # metadata pertaining to the media item
        # - quantitative fields
        'budget': pl.Int64,
        'revenue': pl.Int64,
        'runtime': pl.Int64,
        # - country and production information
        'origin_country': pl.List(pl.Utf8),
        'production_companies': pl.List(pl.Utf8),
        'production_countries': pl.List(pl.Utf8),
        'production_status': pl.Utf8,
        # - language information
        'original_language': pl.Utf8,
        'spoken_languages': pl.List(pl.Utf8),
        # - other string fields
        'genre': pl.List(pl.Utf8),
        'original_media_title': pl.Utf8,
        # - long string fields
        'tagline': pl.Utf8,
        'overview': pl.Utf8,
        # - ratings info
        'tmdb_rating': pl.Float64,
        'tmdb_votes': pl.Int64,
        'rt_score': pl.Int16,
        'metascore': pl.Int16,
        'imdb_rating': pl.Float64,
        'imdb_votes': pl.Int64,
        # timestamps
        'created_at': pl.Datetime,
        'updated_at': pl.Datetime,
    }

    # Common required columns
    required_columns = [
        'imdb_id',
        'media_type',
        'media_title',
        'release_year'
    ]

    def __init__(self, data: Optional[Any] = None):
        """
        Initialize with data that can be converted to a polars DataFrame.

        Args:
            data: Data to convert to DataFrame
        """
        if data is None:
            # Create empty DataFrame with proper schema
            self._df = pl.DataFrame(schema=self.schema)
        elif isinstance(data, pl.DataFrame):
            self._validate_and_prepare(data)
        else:
            # Try to create from other data types (dict, list, etc.)
            try:
                # Use the predefined schema here
                self._df = pl.DataFrame(data, schema=self.schema)
                self._validate_and_prepare(self._df)
            except Exception as e:
                raise ValueError(
                    f"Could not create TrainingDataFrame from data: {e}")

    def _validate_and_prepare(self, df: pl.DataFrame) -> None:
        """
        Validate that DataFrame conforms to the required schema and prepare it.
        """
        # Check required columns
        missing = [col for col in self.required_columns if
                   col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get current timestamp once to ensure consistency
        current_timestamp = datetime.now(timezone.utc)

        # For both timestamp columns, conditionally add or update
        timestamp_exprs = []

        # Handle created_at
        if 'created_at' in df.columns:
            # First check if the column has any non-null values
            if df['created_at'].null_count() < len(df):
                timestamp_exprs.append(
                    pl.when(pl.col('created_at').is_null())
                    .then(pl.lit(current_timestamp))
                    .otherwise(
                        # Only apply timezone conversion to non-null values
                        pl.when(pl.col('created_at').is_not_null())
                        .then(pl.col('created_at').dt.replace_time_zone('UTC'))
                        .otherwise(pl.lit(current_timestamp))
                    )
                    .alias('created_at')
                )
            else:
                # All values are null, just use current timestamp
                timestamp_exprs.append(
                    pl.lit(current_timestamp).alias('created_at'))
        else:
            timestamp_exprs.append(
                pl.lit(current_timestamp).alias('created_at'))

        # Handle updated_at - same logic
        if 'updated_at' in df.columns:
            if df['updated_at'].null_count() < len(df):
                timestamp_exprs.append(
                    pl.when(pl.col('updated_at').is_null())
                    .then(pl.lit(current_timestamp))
                    .otherwise(
                        pl.when(pl.col('updated_at').is_not_null())
                        .then(pl.col('updated_at').dt.replace_time_zone('UTC'))
                        .otherwise(pl.lit(current_timestamp))
                    )
                    .alias('updated_at')
                )
            else:
                timestamp_exprs.append(
                    pl.lit(current_timestamp).alias('updated_at'))
        else:
            timestamp_exprs.append(
                pl.lit(current_timestamp).alias('updated_at'))

        # Apply all expressions at once
        df = df.with_columns(timestamp_exprs)

        # Set the underlying DataFrame
        self._df = df

    def update(self, df: pl.DataFrame):
        """Update internal DataFrame directly."""
        self._validate_and_prepare(df)
        self._df = df

    @property
    def df(self) -> pl.DataFrame:
        """Access the underlying polars DataFrame."""
        return self._df