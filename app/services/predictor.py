# standard library imports
import logging
import json
import os
import re
from typing import List, Union, Dict, Any, Optional

# third party imports
import pandas as pd
import polars as pl
import xgboost as xgb

# custom/local imports
from app.models.media_prediction_input import MediaPredictionInput
from app.services.mlflow_client import MLflowModelLoader
from app.core.config import settings


class XGBMediaPredictor:
    """
    Class for making predictions with trained XGBoost model.
    Handles preprocessing of input data and model inference.
    """

    def __init__(self, model_name: Optional[str] = None, model_version: Optional[str] = None):
        """
        Initialize the predictor with a trained model from MLflow.

        :param model_name: MLflow registered model name (defaults to settings)
        :param model_version: MLflow model version (defaults to latest)
        """
        self.model_name = model_name or settings.REEL_DRIVER_MLFLOW_MODEL
        self.model_version = model_version or settings.REEL_DRIVER_MODEL_VERSION
        
        # Initialize MLflow client
        self.mlflow_loader = MLflowModelLoader()
        
        # Load model and artifacts from MLflow
        self._load_from_mlflow()
        
        # Initialize categorical mappings from schema
        self._init_category_mappings_from_schema()

    def _load_from_mlflow(self):
        """Load model and artifacts from MLflow."""
        logging.info(f"Loading model {self.model_name} version {self.model_version} from MLflow")
        
        try:
            # Download model and artifacts
            artifacts = self.mlflow_loader.download_model_artifacts(
                self.model_name, 
                self.model_version
            )
            
            # Set model
            self.model = artifacts['model']
            
            # Extract feature names from the model
            self.feature_names = self.model.get_booster().feature_names
            
            # Set normalization parameters
            self.normalization = artifacts['normalization']
            
            # Set engineered schema
            self.engineered_schema = artifacts['schema']
            
            # Store metadata
            self.run_id = artifacts['run_id']
            self.loaded_model_version = artifacts['model_version']
            
            logging.info(f"Successfully loaded model with {len(self.feature_names)} features")
            logging.info(f"Normalization parameters loaded for {len(self.normalization)} features")
            logging.info(f"Schema loaded with {len(self.engineered_schema)} categorical mappings")
            
        except Exception as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            raise

    def _init_category_mappings_from_schema(self):
        """
        Initialize mappings for categorical features from engineered schema.
        Uses the schema data from MLflow artifacts instead of extracting from feature names.
        """
        # Initialize empty mappings
        self.genres = []
        self.origin_countries = []
        self.production_countries = []
        self.spoken_languages = []
        
        # Extract mappings from engineered schema
        for schema_item in self.engineered_schema:
            original_column = schema_item['original_column']
            exploded_mapping = schema_item['exploded_mapping']
            
            if original_column == 'genre':
                # Extract genre names by removing prefix
                self.genres = [mapping.replace('genre_', '') for mapping in exploded_mapping]
            elif original_column == 'origin_country':
                self.origin_countries = [mapping.replace('origin_country_', '') for mapping in exploded_mapping]
            elif original_column == 'production_countries':
                # Note: training uses singular 'production_country' prefix
                self.production_countries = [mapping.replace('production_country_', '') for mapping in exploded_mapping]
            elif original_column == 'spoken_languages':
                # Note: training uses singular 'spoken_language' prefix
                self.spoken_languages = [mapping.replace('spoken_language_', '') for mapping in exploded_mapping]
        
        # Log extracted mappings
        logging.debug(f"Extracted {len(self.genres)} genre categories from schema")
        logging.debug(f"Extracted {len(self.origin_countries)} origin country categories from schema")
        logging.debug(f"Extracted {len(self.production_countries)} production country categories from schema")
        logging.debug(f"Extracted {len(self.spoken_languages)} spoken language categories from schema")


    def _encode_list_column(self, column_data: pl.Series, column_name: str, valid_categories: List[str]) -> pl.DataFrame:
        """
        Convert a column of string lists into binary indicator columns.
        Matches the training pipeline encoding logic.
        """
        if column_data[0] is None:
            input_values = []
        else:
            input_values = column_data[0].to_list() if column_data[0] is not None else []

        # Create binary columns for each valid category
        result_columns = []
        for category in valid_categories:
            # Check if category exists in input values
            binary_value = 1 if any(val for val in input_values if val and
                                  re.sub(r'[^\w]', '_', val.lower()) == category) else 0
            result_columns.append(pl.lit(binary_value).alias(f"{column_name}_{category}"))

        return pl.DataFrame().with_columns(result_columns)

    def preprocess(self, media_input: Union[MediaPredictionInput, Dict]) -> pd.DataFrame:
        """
        Preprocess media input for model prediction.
        Follows exact same pipeline as training.
        """
        import logging

        # Convert input to dict if it's a Pydantic model
        if isinstance(media_input, MediaPredictionInput):
            input_dict = media_input.model_dump()
        else:
            input_dict = media_input

        # Create a polars DataFrame with the input
        df = pl.DataFrame([input_dict])
        logging.debug(f"Input data: {df}")

        # Normalize continuous fields using training normalization parameters
        normalization_exprs = []

        continuous_fields = [
            'release_year', 'budget', 'revenue', 'runtime',
            'tmdb_rating', 'tmdb_votes', 'rt_score', 'metascore',
            'imdb_rating', 'imdb_votes'
        ]

        for field in continuous_fields:
            if field in self.normalization:
                norm_min = self.normalization[field]["min"]
                norm_max = self.normalization[field]["max"]

                normalization_exprs.append(
                    pl.when(pl.col(field).is_not_null())
                    .then((pl.col(field) - norm_min) / (norm_max - norm_min))
                    .otherwise(0.0)  # Fill nulls with 0
                    .alias(f"{field}_norm")
                )

        # Apply normalization
        feature_df = df.with_columns(normalization_exprs)

        # drop columns not yet incorporated into training model
        if 'production_companies' in feature_df.columns:
            feature_df = feature_df.drop('production_companies')
        if 'tagline' in feature_df.columns:
            feature_df = feature_df.drop('tagline')
        if 'overview' in feature_df.columns:
            feature_df = feature_df.drop('overview')

        # Encode categorical list features
        # Origin countries
        origin_country_encoded = self._encode_list_column(
            feature_df['origin_country'] if 'origin_country' in feature_df.columns else pl.Series([None]),
            'origin_country',
            self.origin_countries
        )
        feature_df = pl.concat([feature_df, origin_country_encoded], how='horizontal')

        # Production countries (note: singular prefix in training)
        production_countries_encoded = self._encode_list_column(
            feature_df['production_countries'] if 'production_countries' in feature_df.columns else pl.Series([None]),
            'production_country',
            self.production_countries
        )
        feature_df = pl.concat([feature_df, production_countries_encoded], how='horizontal')

        # Spoken languages (note: singular prefix in training)
        spoken_languages_encoded = self._encode_list_column(
            feature_df['spoken_languages'] if 'spoken_languages' in feature_df.columns else pl.Series([None]),
            'spoken_language',
            self.spoken_languages
        )
        feature_df = pl.concat([feature_df, spoken_languages_encoded], how='horizontal')

        # Genres
        genre_encoded = self._encode_list_column(
            feature_df['genre'] if 'genre' in feature_df.columns else pl.Series([None]),
            'genre',
            self.genres
        )
        feature_df = pl.concat([feature_df, genre_encoded], how='horizontal')

        # Handle categorical single-value features (production_status, original_language)
        # These are treated as categorical by pandas in training
        if 'production_status' in feature_df.columns:
            production_status_value = feature_df['production_status'][0]
        else:
            production_status_value = None

        if 'original_language' in feature_df.columns:
            original_language_value = feature_df['original_language'][0]
        else:
            original_language_value = None

        # Add categorical columns
        feature_df = feature_df.with_columns([
            pl.lit(production_status_value).alias('production_status'),
            pl.lit(original_language_value).alias('original_language')
        ])

        # Fill nulls with 0 for all numeric columns
        feature_df = feature_df.fill_null(0)

        # Convert to pandas for XGBoost compatibility
        pandas_df = feature_df.to_pandas()

        # Convert categorical columns to pandas categorical (matching training)
        categorical_cols = ['production_status', 'original_language']
        for col in categorical_cols:
            if col in pandas_df.columns:
                pandas_df[col] = pandas_df[col].astype('category')

        # Ensure all expected features are present and in the right order
        for feature in self.feature_names:
            if feature not in pandas_df.columns:
                if feature in categorical_cols:
                    pandas_df[feature] = pd.Categorical([None])
                else:
                    pandas_df[feature] = 0

        # Reorder columns to match the model's expected features
        pandas_df = pandas_df[self.feature_names]

        logging.debug(f"Preprocessed features: {pandas_df.columns.tolist()}")
        logging.debug(f"Feature dtypes: {pandas_df.dtypes}")

        return pandas_df

    def predict(self, media_input: Union[MediaPredictionInput, Dict]) -> Dict[str, Any]:
        """
        Make a prediction for a media item.

        :param media_input: Media data as MediaPredictionInput or Dict
        :type media_input: Union[MediaPredictionInput, Dict]
        :return: Dict with prediction results (prediction and probability)
        :rtype: Dict[str, Any]
        """
        # Preprocess the input
        features_df = self.preprocess(media_input)
        logging.debug(f"Prediction features shape: {features_df.shape}")

        # Convert numeric columns to proper dtypes
        for col in features_df.columns:
            if col not in ['production_status', 'original_language']:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0.0)

        # Make prediction
        prediction = bool(self.model.predict(features_df)[0])

        # Get probability
        probabilities = self.model.predict_proba(features_df)[0]
        probability = float(probabilities[1])  # Probability of positive class

        # Create result object
        result = {
            "imdb_id": media_input["imdb_id"] if isinstance(media_input, dict) else media_input.imdb_id,
            "prediction": prediction,
            "probability": probability
        }

        logging.debug(f"Prediction result: {result}")
        return result

    def predict_batch(self, media_inputs: List[Union[MediaPredictionInput, Dict]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple media items.

        :param media_inputs: List of media data as MediaPredictionInput or Dict
        :type media_inputs: List[Union[MediaPredictionInput, Dict]]
        :return: List of Dicts with prediction results
        :rtype: List[Dict[str, Any]]
        """
        import logging

        logging.debug(f"Processing batch of {len(media_inputs)} inputs")
        results = []

        for input_item in media_inputs:
            result = self.predict(input_item)
            results.append(result)

        return results