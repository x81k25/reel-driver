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
from app.services.mlflow_client import (
    MLflowModelLoader, 
    MLflowConnectionError, 
    MLflowModelNotFoundError, 
    MLflowArtifactError
)
from app.core.config import settings


class ModelLoadingError(Exception):
    """Raised when model cannot be loaded for prediction."""
    pass


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
        self.model_version = model_version or settings.REEL_DRIVER_API_MODEL_VERSION
        
        try:
            # Initialize MLflow client
            self.mlflow_loader = MLflowModelLoader()
            
            # Load model and artifacts from MLflow
            self._load_from_mlflow()
            
            # Initialize categorical mappings from schema
            self._init_category_mappings_from_schema()
            
        except (MLflowConnectionError, MLflowModelNotFoundError, MLflowArtifactError) as e:
            logging.error(f"Failed to initialize predictor: {e}")
            raise ModelLoadingError(f"Failed to initialize predictor: {e}")
        except Exception as e:
            logging.error(f"Unexpected error initializing predictor: {e}")
            raise ModelLoadingError(f"Unexpected error initializing predictor: {e}")

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
            
        except (MLflowConnectionError, MLflowModelNotFoundError, MLflowArtifactError) as e:
            logging.error(f"Failed to load model from MLflow: {e}")
            raise  # Re-raise to preserve specific error type
        except Exception as e:
            logging.error(f"Unexpected error loading model from MLflow: {e}")
            raise ModelLoadingError(f"Unexpected error loading model from MLflow: {e}")

    def _init_category_mappings_from_schema(self):
        """
        Initialize mappings for categorical features from engineered schema.
        Uses the exact column names from training to ensure dynamic compatibility.
        """
        import re
        
        # Initialize mapping dictionary to store column_name -> (exact_column_names, category_mapping)
        self.column_mappings = {}
        
        # Initialize legacy attributes for backward compatibility
        self.genres = []
        self.origin_countries = []
        self.production_countries = []
        self.spoken_languages = []
        
        # Extract mappings from engineered schema - store exact column names from training
        for schema_item in self.engineered_schema:
            original_column = schema_item['original_column']
            exploded_mapping_raw = schema_item['exploded_mapping']
            
            # Parse the exploded_mapping - it's stored as a string representation of numpy array
            if isinstance(exploded_mapping_raw, str):
                # Remove brackets and split by whitespace, then clean up quotes
                exploded_mapping_clean = exploded_mapping_raw.strip("[]'\"")
                # Split by whitespace and remove quotes from each element
                exploded_mapping = [col.strip("'\"") for col in exploded_mapping_clean.split() if col.strip("'\"")]
            else:
                exploded_mapping = exploded_mapping_raw  # Already a list
            
            # Create a mapping from category value to column name for efficient lookup
            category_to_column = {}
            for column_name in exploded_mapping:
                # Extract the category from the column name by removing the prefix
                # Use dynamic prefix removal based on the original column name
                prefix = original_column + '_'
                if column_name.startswith(prefix):
                    category = column_name[len(prefix):]
                    category_to_column[category] = column_name
                else:
                    # Handle edge case where prefix doesn't match exactly
                    # Extract everything after the first underscore
                    if '_' in column_name:
                        category = column_name.split('_', 1)[1]
                        category_to_column[category] = column_name
            
            # Store both the exact column names and the category mapping
            self.column_mappings[original_column] = {
                'column_names': exploded_mapping,
                'category_to_column': category_to_column
            }
            
            # Set legacy attributes for backward compatibility
            if original_column == 'genre':
                self.genres = exploded_mapping
            elif original_column == 'origin_country':
                self.origin_countries = exploded_mapping
            elif original_column == 'production_countries':
                self.production_countries = exploded_mapping
            elif original_column == 'spoken_languages':
                self.spoken_languages = exploded_mapping
        
        # Log extracted mappings
        logging.debug(f"Schema mappings initialized for {len(self.column_mappings)} categorical columns")
        for orig_col, mapping in self.column_mappings.items():
            logging.debug(f"  {orig_col}: {len(mapping['column_names'])} columns")


    def _encode_list_column(self, column_data: pl.Series, column_name: str) -> pl.DataFrame:
        """
        Convert a column of string lists into binary indicator columns.
        Uses the exact schema from training to ensure dynamic compatibility.
        """
        import re
        
        # Get input values from the series
        if column_data[0] is None or len(column_data) == 0:
            input_values = []
        else:
            input_values = column_data[0].to_list() if column_data[0] is not None else []

        # Get the schema mapping for this column
        if column_name not in self.column_mappings:
            # If column not in schema, return empty DataFrame with no columns
            logging.warning(f"Column {column_name} not found in engineered schema, skipping")
            return pl.DataFrame()
        
        mapping = self.column_mappings[column_name]
        category_to_column = mapping['category_to_column']
        
        # Apply the same cleaning function used in training
        def clean_value(value: str) -> str:
            """Apply the same cleaning logic as training feature engineering."""
            # Replace non-word characters with underscores and convert to lowercase
            cleaned = re.sub(r'[^\w]', '_', value.lower())
            # Replace multiple consecutive underscores with single underscore
            cleaned = re.sub(r'_+', '_', cleaned)
            # Remove leading/trailing underscores
            return cleaned.strip('_')

        # Create binary columns for each category defined in the schema
        result_columns = []
        for category, exact_column_name in category_to_column.items():
            # Check if any input value matches this category (after cleaning)
            binary_value = 0
            for input_val in input_values:
                if input_val and clean_value(input_val) == category:
                    binary_value = 1
                    break
            
            # Use the exact column name from training schema
            result_columns.append(pl.lit(binary_value).alias(exact_column_name))

        # Handle case where no categories were matched but we still need to return the schema columns
        if not result_columns:
            # Create all schema columns with 0 values
            for exact_column_name in mapping['column_names']:
                result_columns.append(pl.lit(0).alias(exact_column_name))

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
            # Only normalize fields that exist in both the input and normalization parameters
            if field in self.normalization and field in df.columns:
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

        # Encode categorical list features dynamically based on engineered_schema
        for original_column in self.column_mappings.keys():
            # Get the column data (or create empty series if column doesn't exist)
            column_data = (
                feature_df[original_column] 
                if original_column in feature_df.columns 
                else pl.Series([None])
            )
            
            # Encode the column using the schema
            encoded_df = self._encode_list_column(column_data, original_column)
            
            # Concatenate the encoded columns to the feature DataFrame
            if len(encoded_df.columns) > 0:  # Only concat if we got columns back
                feature_df = pl.concat([feature_df, encoded_df], how='horizontal')

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

        try:
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
        except Exception as e:
            logging.error(f"Model prediction failed: {e}")
            raise ModelLoadingError(f"Model prediction failed: {e}")

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
            try:
                result = self.predict(input_item)
                results.append(result)
            except Exception as e:
                logging.error(f"Batch prediction failed for item: {e}")
                raise ModelLoadingError(f"Batch prediction failed: {e}")

        return results

    def ensure_latest_model(self):
        """
        Check if current model is latest version and reload if needed.
        If version check fails for any reason, continue with current model.
        """
        try:
            latest_version = self.mlflow_loader.get_latest_model_version(self.model_name)
            if latest_version != self.loaded_model_version:
                logging.info(f"New model version {latest_version} detected (current: {self.loaded_model_version}), reloading...")
                
                # Store current state in case reload fails
                old_model = self.model
                old_version = self.loaded_model_version
                old_features = self.feature_names
                old_normalization = self.normalization
                old_schema = self.engineered_schema
                old_run_id = self.run_id
                old_mappings = self.column_mappings
                
                try:
                    # Reload model and artifacts from MLflow
                    self._load_from_mlflow()
                    self._init_category_mappings_from_schema()
                    logging.info(f"Model successfully updated to version {self.loaded_model_version}")
                except Exception as reload_error:
                    # Restore previous state if reload fails
                    logging.error(f"Model reload failed, reverting to previous version: {reload_error}")
                    self.model = old_model
                    self.loaded_model_version = old_version
                    self.feature_names = old_features
                    self.normalization = old_normalization
                    self.engineered_schema = old_schema
                    self.run_id = old_run_id
                    self.column_mappings = old_mappings
                    
        except Exception as e:
            logging.warning(f"Version check failed, continuing with current model version {self.loaded_model_version}: {e}")