# standard library imports
import logging

# custom/local imports
from src.data_models.media_prediction_input import MediaPredictionInput
from src.inference import XGBMediaPredictor

# log config
# Set up logging to see debug messages
logging.basicConfig(
	level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_predictor():
	xgb_predictor = XGBMediaPredictor()

	# Print the extracted mappings to verify
	print("Genre categories:", xgb_predictor.genres)
	print("Language categories:", xgb_predictor.languages)
	print("Numeric features:", xgb_predictor.numeric_features)

	# test preprocessing
	test_input = MediaPredictionInput(
		hash="test123",
		release_year=2010,
		genre=["Drama", "Comedy", "Action"],
		language=["en", "fr"],
		metascore=75,
		rt_score=85,
		imdb_rating=7.5,
		imdb_votes=10000
	)

	# Test preprocessing
	processed_features = xgb_predictor.preprocess(test_input)
	print("\nProcessed features shape:", processed_features.shape)
	print("\nFirst few columns:", processed_features.columns[:5])
	print("\nFirst row:", processed_features.iloc[0])

	# Test the prediction
	prediction_result = xgb_predictor.predict(test_input)
	print("\nPrediction result:")
	print(f"Would watch: {prediction_result['would_watch']}")
	print(f"Confidence: {prediction_result['probability']:.4f}")

	# Try another example with different values
	another_input = MediaPredictionInput(
		hash="test456",
		release_year=2020,
		genre=["Horror", "Thriller"],
		language=["en"],
		metascore=45,
		rt_score=30,
		imdb_rating=4.8,
		imdb_votes=5000
	)

	another_result = xgb_predictor.predict(another_input)
	print("\nAnother prediction:")
	print(f"Would watch: {another_result['would_watch']}")
	print(f"Confidence: {another_result['probability']:.4f}")

	# Test batch prediction
	batch_inputs = [test_input, another_input]
	batch_results = xgb_predictor.predict_batch(batch_inputs)

	print("\nBatch prediction results:")
	for i, result in enumerate(batch_results):
		print(f"Item {i+1}: Would watch: {result['would_watch']}, Confidence: {result['probability']:.4f}")