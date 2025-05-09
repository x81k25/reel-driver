# standard library imports
import logging
import json
import os
from typing import List, Union, Dict, Any

# third party imports
import pandas as pd
import xgboost as xgb

# custom/local imports
from app.models.media_prediction_input import MediaPredictionInput


class XGBMediaPredictor:
    """
    Class for making predictions with trained XGBoost model.
    Handles preprocessing of input data and model inference.
    """

    def __init__(self, artifacts_path: str = "./model_artifacts/"):
        """
        Initialize the predictor with a trained model and normalization parameters.

        :param model_path: Path to the directory containing model artifacts
        :type model_path: str
        """
        # Load the trained XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(artifacts_path, "xgb_model.json"))

        # Extract feature names from the model
        self.feature_names = self.model.get_booster().feature_names

        # Load normalization parameters
        with open(os.path.join(artifacts_path, "normalization.json"), 'r') as f:
            self.normalization = json.load(f)

        # Initialize mappings for categorical features
        self._init_category_mappings()


    def _init_category_mappings(self):
        """
        Initialize mappings for categorical features based on model features.
        Extract valid genre and language values from model feature names.
        """
        # Extract genre and language categories from feature names
        self.genres = []
        self.languages = []

        for feature in self.feature_names:
            if feature.startswith('genre_'):
                genre = feature.replace('genre_', '')
                self.genres.append(genre)
            elif feature.startswith('language_'):
                language = feature.replace('language_', '')
                self.languages.append(language)

        # We also need to identify numeric features for normalization
        self.numeric_features = [
            feature for feature in self.feature_names
            if feature.endswith('_norm') and not feature.startswith(
                'genre_') and not feature.startswith('language_')
        ]

        # Log extracted information
        logging.debug(f"Identified {len(self.genres)} genre categories")
        logging.debug(f"Identified {len(self.languages)} language categories")
        logging.debug(
            f"Identified {len(self.numeric_features)} numeric features")

    def preprocess(self, media_input: Union[
        MediaPredictionInput, Dict]) -> pd.DataFrame:
        """
        Preprocess media input for model prediction.
        """
        import logging
        import polars as pl

        # Convert input to dict if it's a Pydantic model
        if isinstance(media_input, MediaPredictionInput):
            input_dict = media_input.model_dump()
        else:
            input_dict = media_input

        # Create a polars DataFrame with the input
        df = pl.DataFrame([input_dict])
        logging.debug(f"Input data: {df}")

        # Create feature DataFrame with normalized values
        feature_df = df.with_columns(
            metascore_norm=pl.when(pl.col('metascore').is_not_null())
                .then((pl.col('metascore') - self.normalization["metascore"]["min"]) /
                    (self.normalization["metascore"]["max"] - self.normalization["metascore"]["min"]))
                .otherwise(pl.col('metascore')),
            rt_score_norm=pl.when(pl.col('rt_score').is_not_null())
                .then((pl.col('rt_score') - self.normalization["rt_score"]["min"]) /
                      (self.normalization["rt_score"]["max"] - self.normalization["rt_score"]["min"]))
                .otherwise(pl.col('rt_score')),
            imdb_rating_norm=pl.when(pl.col('imdb_rating').is_not_null())
                .then((pl.col('imdb_rating') - self.normalization["imdb_rating"]["min"]) /
                      (self.normalization["imdb_rating"]["max"] - self.normalization["imdb_rating"]["min"]))
                .otherwise(pl.col('imdb_rating')),
            imdb_votes_norm=pl.when(pl.col('imdb_votes').is_not_null())
                .then((pl.col('imdb_votes') - self.normalization["imdb_votes"]["min"]) /
                      (self.normalization["imdb_votes"]["max"] - self.normalization["imdb_votes"]["min"]))
                .otherwise(pl.col('imdb_votes')),
            release_year_norm=pl.when(pl.col('release_year').is_not_null())
                .then((pl.col('release_year') - self.normalization["release_year"]["min"]) /
                      (self.normalization["release_year"]["max"] - self.normalization["release_year"]["min"]))
                .otherwise(pl.col('release_year'))
        )

        # Process genre features - one-hot encoding
        input_genres = df['genre'][0].to_list() if 'genre' in df.columns and df['genre'][0] is not None else []

        genre_exprs = {
            f"genre_{genre}": pl.lit(1 if genre in input_genres else 0)
            for genre in self.genres
        }

        # Process language features - one-hot encoding
        input_languages = df['language'][0].to_list() if 'language' in df.columns and df['language'][0] is not None else []

        language_exprs = {
            f"language_{lang}": pl.lit(
                1 if input_languages and lang in input_languages else 0)
            for lang in self.languages
        }

        # Add all categorical features
        feature_df = feature_df.with_columns(**genre_exprs).with_columns(
            **language_exprs)

        # Fill nulls with 0
        feature_df = feature_df.fill_null(0)

        # Convert to pandas for XGBoost compatibility
        pandas_df = feature_df.to_pandas()

        # Ensure all expected features are present and in the right order
        for feature in self.feature_names:
            if feature not in pandas_df.columns:
                pandas_df[feature] = 0

        # Reorder columns to match the model's expected features
        pandas_df = pandas_df[self.feature_names]

        logging.debug(f"Preprocessed features: {pandas_df.columns.tolist()}")
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

        # Make prediction
        prediction = bool(self.model.predict(features_df)[0])

        # Get probability
        probabilities = self.model.predict_proba(features_df)[0]
        probability = float(probabilities[1])  # Probability of positive class

        # Create result object
        result = {
            "hash": media_input["hash"] if isinstance(media_input, dict) else media_input.hash,
            "prediction": prediction,
            "probability": probability,
            "would_watch": prediction
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