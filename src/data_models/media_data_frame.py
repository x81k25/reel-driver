from datetime import datetime, timezone
from enum import Enum
import polars as pl
from typing import List, Dict, Any, Optional, ClassVar, Type

# ------------------------------------------------------------------------------
# enum classes
# ------------------------------------------------------------------------------

pl.enable_string_cache()

class PipelineStatus(str, Enum):
    INGESTED = 'ingested'
    PAUSED = 'paused'
    PARSED = 'parsed'
    METADATA_COLLECTED = 'metadata_collected'
    REJECTED = 'rejected'
    QUEUED = 'queued'
    DOWNLOADING = 'downloading'
    DOWNLOADED = 'downloaded'
    TRANSFERRED = 'transferred'
    COMPLETE = 'complete'


class RejectionStatus(str, Enum):
    UNFILTERED = 'unfiltered'
    ACCEPTED = 'accepted'
    FAILED = 'failed'
    OVERRIDE = 'override'


class MediaType(str, Enum):
    MOVIE = 'movie'
    TV_SHOW = 'tv_show'
    TV_SEASON = 'tv_season'


class RssSource(str, Enum):
    YTS = 'yts.mx'
    EPISODE_FEED = 'episodefeed.com'


# ------------------------------------------------------------------------------
# unified media dataframe class
# ------------------------------------------------------------------------------

class MediaDataFrame:
    """Unified rigid polars DataFrame for all media types matching SQL schemas."""

    # Complete schema with all possible fields
    schema = {
        # identifier column
        'hash': pl.Utf8,
        # media information
        'media_type': pl.Categorical,
        'media_title': pl.Utf8,
        'season': pl.Int64,
        'episode': pl.Int64,
        'release_year': pl.Int64,
        # pipeline status information
        'pipeline_status': pl.Categorical,
        'error_status': pl.Boolean,
        'error_condition': pl.Utf8,
        'rejection_status': pl.Categorical,
        'rejection_reason': pl.Utf8,
        # path information
        'parent_path': pl.Utf8,
        'target_path': pl.Utf8,
        # download information
        'original_title': pl.Utf8,
        'original_path': pl.Utf8,
        'original_link': pl.Utf8,
        'rss_source': pl.Categorical,
        'uploader': pl.Utf8,
        # metadata pertaining to the media item
        'imdb_id': pl.Utf8,
        'tmdb_id': pl.Int64,
        'genre': pl.List(pl.Utf8),
        'language': pl.List(pl.Utf8),
        'rt_score': pl.Int64,
        'metascore': pl.Int64,
        'imdb_rating': pl.Float64,
        'imdb_votes': pl.Int64,
        # metadata pertaining to the video file
        'resolution': pl.Utf8,
        'video_codec': pl.Utf8,
        'upload_type': pl.Utf8,
        'audio_codec': pl.Utf8,
        # timestamps
        'created_at': pl.Datetime,
        'updated_at': pl.Datetime,
    }

    # Common required columns
    required_columns = [
        'hash',
        'original_title',
        'media_type',
        'pipeline_status',
        'error_status',
        'rejection_status'
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
                    f"Could not create MediaDataFrame from data: {e}")


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

# ------------------------------------------------------------------------------
# end of data_models.py
# ------------------------------------------------------------------------------