# internal libary imports
import os

# third-party imports
from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# ------------------------------------------------------------------------------
# build and train model
# ------------------------------------------------------------------------------

load_dotenv()

# connect to k8s mlflow service
mlflow_uri = "http://" + os.getenv('MLFLOW_HOST') + ":" + os.getenv('MLFLOW_PORT')

mlflow.set_tracking_uri(mlflow_uri)

# Set experiment
mlflow.set_experiment("reel_driver")

# ------------------------------------------------------------------------------
# build and train model
# ------------------------------------------------------------------------------

# read in training data
df = pd.read_parquet('./data/binomial_classifier_training_data.parquet')
df.set_index('hash', inplace=True)

# drop media_title during training
df.drop('media_title', axis=1, inplace=True)

# convert values to numpy for xgboost ingestion
X = df.drop('label', axis=1)

y = df['label']

# train/test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
	X,
	y,
	test_size=0.15,
	random_state=42
)

# use val percentage of 15% of the original size of the data
val_size = (.15)/(1-0.15)

# split out validation data
X_train, X_val, y_train, y_val = train_test_split(
	X_train_val,
	y_train_val,
	test_size=val_size,
	random_state=42,
)

model = xgb.XGBClassifier(
	objective='binary:logistic',
	n_estimators=100
)

# Define parameter grid
param_grid = {
   'max_depth': [3, 5, 7],
   'learning_rate': [0.01, 0.1, 0.2],
   'n_estimators': [50, 100, 200],
   'subsample': [0.8, 1.0],
   'colsample_bytree': [0.8, 1.0],
   'gamma': [0, 0.1, 0.2]
}

# Start MLflow run
with mlflow.start_run(run_name="xgboost_grid_search"):
	# training notes
	mlflow.set_tag("notes", "cleaned false positives")

	# Log data info
	mlflow.log_param("train_size", X_train.shape[0])
	mlflow.log_param("val_size", X_val.shape[0])
	mlflow.log_param("test_size", X_test.shape[0])
	mlflow.log_param("features", list(X.columns))

	# Grid search setup
	model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
	grid_search = GridSearchCV(
		estimator=model,
		param_grid=param_grid,
		cv=5,
		scoring='recall',
		verbose=1,
		n_jobs=-1
	)

	# Fit model
	grid_search.fit(X_train, y_train)
	model = grid_search.best_estimator_

	# Log best parameters
	for param, value in grid_search.best_params_.items():
		mlflow.log_param(param, value)

	# Validation metrics
	val_score = model.score(X_val, y_val)
	mlflow.log_metric("validation_accuracy", val_score)

	# Test metrics
	y_test_pred = model.predict(X_test)
	y_test_pred_proba = model.predict_proba(X_test)[:, 1]

	# Calculate metrics
	tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
	accuracy = accuracy_score(y_test, y_test_pred)
	precision = precision_score(y_test, y_test_pred)
	recall = recall_score(y_test, y_test_pred)
	f1 = f1_score(y_test, y_test_pred)
	auc = roc_auc_score(y_test, y_test_pred_proba)

	# Log metrics
	mlflow.log_metric("accuracy", accuracy)
	mlflow.log_metric("precision", precision)
	mlflow.log_metric("recall", recall)
	mlflow.log_metric("f1", f1)
	mlflow.log_metric("auc", auc)
	mlflow.log_metric("true_positives", tp)
	mlflow.log_metric("false_positives", fp)
	mlflow.log_metric("true_negatives", tn)
	mlflow.log_metric("false_negatives", fn)

	# Print results
	print(f"Validation score: {val_score}")
	print(f"Best params: {grid_search.best_params_}")
	print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_test_pred)}")
	print(f"Accuracy: {accuracy:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1 Score: {f1:.4f}")
	print(f"AUC-ROC: {auc:.4f}")

	# Create results dataframe
	y_pred = model.predict(X)
	y_pred_proba = model.predict_proba(X)[:, 1]
	results_df = pd.DataFrame({
		'actual': y,
		'predicted': y_pred,
		'probability': y_pred_proba
	}, index=y.index)

	# Save results
	results_path = "./data/binomial_classifier_results.parquet"
	results_df.to_parquet(results_path)

# ------------------------------------------------------------------------------
# end of binomial_classifier.py
# ------------------------------------------------------------------------------
