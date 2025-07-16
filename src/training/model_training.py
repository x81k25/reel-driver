# internal library imports
import json
import os
import re

# third-party imports
from dotenv import load_dotenv
from loguru import logger
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

# custom/internal imports
import src.utils as utils

# ------------------------------------------------------------------------
# supporting functions
# ------------------------------------------------------------------------

def unpack_engineered(
	engineered: pl.DataFrame,
	engineered_schema: pl.DataFrame
) -> pl.DataFrame:
	"""
	flattens out all array columns

	:param engineered: engineered DataFrame with categorical list columns
		stored as sparse array
	:param engineered_schema: column mapping of exactly how to unpack the
		sparse array columns
	:return: DataFrame containing the unpacked columns
	"""
	# Get the column names and convert list column to struct, then unnest
	engineered_unpacked = engineered.clone()

	for col in engineered_schema['original_column']:
		column_names = (engineered_schema.filter(
			pl.col('original_column') == col)['exploded_mapping']
				.item()
				.to_list()
		)

		if len(column_names) > 1:
			engineered_unpacked = (
				engineered_unpacked.with_columns(
					pl.col(col)
						.list
						.to_struct(fields=column_names)
						.alias(col + "_struct")
				).unnest(col + "_struct")
				.drop(col)
			)

	return engineered_unpacked


def prep_engineered(
	df: pl.DataFrame
) -> pd.DataFrame:
	"""
	performs all other processing functions on the dataframe before model
		ingestion

	:param df: engineered DataFrame before final pre-processing
	:return: DataFrame ready for model ingestion
	"""
	# Get the column names and convert list column to struct, then unnest
	df_prepped = df.clone()

	# drop unneeded columns
	df_prepped = df_prepped.drop([
		'media_title',
		'created_at',
		'updated_at'
	])

	pdf_prepped = df_prepped.to_pandas()
	pdf_prepped.set_index('imdb_id', inplace=True)

	# convert categoricals
	categorical_cols = ['production_status', 'original_language']
	for col in categorical_cols:
		pdf_prepped[col] = pdf_prepped[col].astype('category')

	# convert numerics
	num_cols = ['imdb_rating', 'tmdb_rating']
	for col in num_cols:
		pdf_prepped[col] = pdf_prepped[col].astype(float)

	return pdf_prepped


def data_split(
	df: pd.DataFrame,
	split_size: float = 0.2,
	random_state: int = 42
) -> tuple[
	pd.DataFrame,
	pd.DataFrame,
	pd.DataFrame,
	pd.Series,
	pd.Series,
	pd.Series
]:
	"""
	take full df and performs test/train/val split based off of size param

	:param df: whole dataset including the label column
	:param split_size: the size in decimal percentage of the training and
		the validation samples
	:param random_state: random state param for reproducibility
	:return: tuple containing[
		the full X DataFrame
		X_train DataFrame
		X_test DataFrame
		the full y Series
		y_train Series
		y_test Series
	]
	"""
	df_split = df.copy()

	# features/label split
	X = df_split.drop('label', axis=1)
	y = df_split['label']

	# train/test split
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=split_size,
		random_state=random_state
	)

	logger.info(f"split complete on {len(X)} data elements with {len(X.columns)} features")
	logger.info(f"{len(X_train)} elements to be used for training")
	logger.info(f"{len(X_test)} elements to be used for testing")

	return (
		X,
		X_train,
		X_test,
		y,
		y_train,
		y_test
	)


def track_input_metrics(
	X: pd.DataFrame,
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
):
	"""
	logs all metrics recorded by MLFlow before the model is executed

	:param X: full feature set
	:param X_train: feature training set
	:param X_test:  feature test set
	:return: None
	"""
	# Log data info
	mlflow.log_param("train_size", X_train.shape[0])
	mlflow.log_param("test_size", X_test.shape[0])
	mlflow.log_param("features", list(X.columns))

	return


def track_output_metrics(
	hyper_search,
	X_test: pd.DataFrame,
	y_test: pd.Series
):
	"""
	generates all relevant model output metrics and logs in MLflow

	:param hyper_search: full hyperparameters search object
	:param X_test: test features
	:param y_test: label output subset
	:return: None
	"""
	# store the hyper-param search results themselves
	# hyper_search_results = hyper_search.cv_results_
	#
	# hyper_search_results_formatted = {}
	#
	# for i in range(len(hyper_search_results['rank_test_score'])):
	# 	model_run = {}
	# 	# grab the i element of all key value paries
	# 	for key, value in hyper_search_results.items():
	# 		# remove duplicate param values that are already contained in params object
	# 		if not re.search(r'param_', key):
	# 			model_run[key] = value[i]
	# 		# move the split test scores into a single list object
	# 		split_test_scores = []
	# 		for key, value in model_run.items():
	# 			if re.search(r'split\d+_test_score', key):
	# 				split_test_scores.append(value)
	# 		model_run['split_test_scores'] = split_test_scores
	# 		# remove now duplicate individual split test score items
	# 		model_run_clean = {}
	# 		for key, value in model_run.items():
	# 			if not re.search(r'split\d+_test_score', key):
	# 				model_run_clean[key] = value
	# 	# add individual run to aggregate dict
	# 	hyper_search_results_formatted[f"model_run_{i}"] = model_run_clean
	#
	# # sort by rank score
	# sorted_keys = sorted(
	# 	[key for key in hyper_search_results_formatted.keys() if
	# 	 key.startswith('model_run_')],
	# 	key=lambda x: hyper_search_results_formatted[x]['rank_test_score'],
	# )
	#
	# hyper_search_results_ranked = {}
	#
	# for key in sorted_keys:
	# 	hyper_search_results_ranked[key] = hyper_search_results_formatted[key]

	hyper_search_results = hyper_search.cv_results_

	hyper_search_results_formatted = {}

	for i in range(len(hyper_search_results['rank_test_score'])):
		model_run = {}
		split_test_score = []

		for key, value in hyper_search_results.items():
			# skip param keys, as they are already included in params{}
			if re.search(r'param_', key):
				continue







	mlflow.log_dict(hyper_search_results_ranked, "hyper_search_results.json")

	# Log best parameters
	for param, value in hyper_search.best_params_.items():
		mlflow.log_param(param, value)
	logger.info(f"Best params: {hyper_search.best_params_}")

	# generate est model from hyper_search
	best_model = hyper_search.best_estimator_

	# generate predictions
	y_pred = best_model.predict(X_test)
	y_pred_proba = best_model.predict_proba(X_test)[:, 1]

	# Calculate metrics
	if len(np.unique(y_test)) > 1:
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
		mlflow.log_metric("true_positives", tp)
		mlflow.log_metric("false_positives", fp)
		mlflow.log_metric("true_negatives", tn)
		mlflow.log_metric("false_negatives", fn)
		logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

		if (tp + fp > 0) and (tp + fn > 0):
			f1 = f1_score(y_test, y_pred)
			mlflow.log_metric("f1", f1)
			logger.info(f"F1 Score: {f1:.4f}")

		accuracy = accuracy_score(y_test, y_pred)
		mlflow.log_metric("accuracy", accuracy)
		logger.info(f"Accuracy: {accuracy:.4f}")

		if tp + fp > 0:
			precision = precision_score(y_test, y_pred)
			mlflow.log_metric("precision", precision)
			logger.info(f"Precision: {precision:.4f}")

		if tp + fn > 0:
			recall = recall_score(y_test, y_pred)
			mlflow.log_metric("recall", recall)
			logger.info(f"Recall: {recall:.4f}")

		if len(np.unique(y_test)) > 1:
			auc = roc_auc_score(y_test, y_pred_proba)
			mlflow.log_metric("auc", auc)
			logger.info(f"AUC-ROC: {auc:.4f}")

	return

# ------------------------------------------------------------------------
# primary function
# ------------------------------------------------------------------------

def xgb_hyp_op(
	search_strategy: str = 'grid',
	random_n_iter: int = 5,
	random_seed: int = 42
):
	"""
	end-to-end stateless hyperparameter optimization service

	:param search_strategy: either 'random' or 'grid'
	:param random_n_iter: number of iterations if using random search
	:param random_state: random_state var to be passed to all functions
	:return: None
	"""
	# load dotenv at the module level if running locally
	if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
		load_dotenv(override=True)
		search_strategy = 'random'
		random_n_iter = 5
		random_seed = 42

	# set minio env vars
	os.environ['MLFLOW_S3_ENDPOINT_URL'] = str(
		os.environ['REEL_DRIVER_MINIO_ENDPOINT'] +
		":" +
		os.environ['REEL_DRIVER_MINIO_PORT']
	)

	os.environ['AWS_ACCESS_KEY_ID'] = os.environ['REEL_DRIVER_MINIO_ACCESS_KEY']
	os.environ['AWS_SECRET_ACCESS_KEY'] = os.environ['REEL_DRIVER_MINIO_SECRET_KEY']

	# set random seeds
	np.random.seed(random_seed)

	# connect to k8s mlflow service
	mlflow_uri = "http://" + os.getenv('REEL_DRIVER_MLFLOW_HOST') + ":" + os.getenv('REEL_DRIVER_MLFLOW_PORT')
	mlflow.set_tracking_uri(mlflow_uri)

	# Set experiment
	mlflow.set_experiment(experiment_name=os.getenv('REEL_DRIVER_MLFLOW_EXPERIMENT'))

	logger.info("extracting engineered features")

	# read in training data and metadata
	engineered = utils.select_star(table="engineered")
	engineered_schema = utils.select_star(table="engineered_schema")
	normalization_table = utils.select_star(table="engineered_normalization_table")

	# perform prep on engineered data
	df = unpack_engineered(
		engineered=engineered,
		engineered_schema=engineered_schema
	)
	pdf = prep_engineered(df=df)

	# split
	X, X_train, X_test, y, y_train, y_test = data_split(pdf, random_seed)

	# Start MLflow run
	with mlflow.start_run(run_name="xgboost_grid_search"):

		# if running locally enter training notes
		if os.getenv("LOCAL_DEVELOPMENT", '') == "true":
			notes = input("Enter notes for this model run: ")
		else:
			notes = "automated training run"

		mlflow.set_tag("notes", notes)

		# store all input model values in MLFlow
		track_input_metrics(X, X_train, X_test)

		# define model
		model = xgb.XGBClassifier(
			objective='binary:logistic',
			enable_categorical=True
		)

		# Define parameter grid
		param_grid = {
			'scale_pos_weight': [1, 5, 9, 15],
			'max_depth': [3, 5, 7],
			'n_estimators': [50, 100, 200],
			'min_child_weight': [1 ,3, 5],
			'learning_rate': [0.01, 0.1, 0.2],
			'gamma': [0, 0.1, 0.2],
			'reg_alpha': [0, 0.01, 0.1, 1.0],
			'subsample': [0.8, 1.0],
			'max_delta_step': [0, 1, 5],
			'colsample_bytree': [0.8, 1.0],
			'colsample_bylevel': [0.8, 1.0],
			'random_state': [42],
			'enable_categorical': [True]
		}

		# select search strategy
		if search_strategy == "grid":
			hyper_search = GridSearchCV(
				estimator=model,
				param_grid=param_grid,
				cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed),
				scoring='f1',
				verbose=3,
				n_jobs=8
			)
		elif search_strategy == "random":
			hyper_search = RandomizedSearchCV(
			   estimator=model,
			   param_distributions=param_grid,
			   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed),
			   verbose=3,
			   n_jobs=8,
			   n_iter=random_n_iter
			)

		# Fit model
		hyper_search.fit(X_train, y_train)

		# track all output metrics
		track_output_metrics(
			hyper_search=hyper_search,
			X_test=X_test,
			y_test=y_test
		)

		logger.info('building full_model')

		# generate model with best params and train with all data
		best_params = hyper_search.best_params_
		full_model = xgb.XGBClassifier(**best_params)
		full_model.fit(X, y)

		# Use a small sample for signature inference
		signature = infer_signature(
			X.head(10),
			full_model.predict(X.head(10))
		)

		mlflow.sklearn.log_model(
			sk_model=full_model,
			artifact_path="model",
			registered_model_name=os.getenv("REEL_DRIVER_MLFLOW_MODEL"),
			signature=signature,
			conda_env=None
		)

		logger.info('full_model stored in MLflow model registry')

		# Save normalization table and engineered schema as JSON artifacts (in-memory)
		mlflow.log_dict(
			normalization_table.to_pandas().to_dict(orient='records'),
			"model-artifacts/engineered_normalization_table.json"
		)
		
		mlflow.log_dict(
			engineered_schema.to_pandas().to_dict(orient='records'),
			"model-artifacts/engineered_schema.json"
		)

		logger.info('normalization table and engineered schema saved as MLflow artifacts')

# main guard
def __main__():
	xgb_hyp_op()

# -----------------------------------------------------------------------------
# end of _03_xgb_binomial_classifier_build_model.py
# -----------------------------------------------------------------------------