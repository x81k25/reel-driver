# Automatic Transmission Algorithm

A personal media curation algorithm trained on my personally labeled data.

## Overview

This repository contains a machine learning solution that determines which media to add to my personal library. The project uses XGBoost to train a binary classifier that predicts whether I would want to keep specific media based on features including ratings (IMDB, Rotten Tomatoes, Metascore), release year, genre, and language.

Currently focused on movies, with plans to extend to TV shows and other media types.

## Project Structure

```
automatic-transmission-algo/
├── data/                  # Data files (not included in repo due to size)
│   ├── media.parquet
│   ├── binomial_classifier_training_data.parquet
│   ├── binomial_classifier_results.parquet
│   ├── false_positives.json
│   └── false_negatives.json
├── notebooks/             # Jupyter notebooks for analysis
│   └── binomial_classifier_analysis.ipynb
├── scripts/               # Python scripts for data processing and model training
│   ├── _00_error_correction.py      # Ad hoc error correction
│   ├── _01_prepare_training_data.py # Data preparation (step 1)
│   └── _02_binomial_classifier.py   # Model training (step 2)
├── .gitignore
├── README.md
├── requirements.in        # Input file for dependency management
└── requirements.txt       # Generated dependencies with pinned versions
```

## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL database with appropriate schema setup
- MLflow server (for model tracking)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PG_DB=your_database_name
PG_USER=your_database_user
PG_PASS=your_database_password
PG_HOST=your_database_host
PG_PORT=your_database_port
MLFLOW_HOST=your_mlflow_host
MLFLOW_PORT=your_mlflow_port
```

### Installation

This project uses `uv` for dependency management:

```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate

# Install dependencies using uv
uv pip install -r requirements.txt
```

## Workflow

The project follows this workflow:

1. **Data Preparation** (`_01_prepare_training_data.py`): 
   - Extracts data from PostgreSQL database
   - Transforms and cleans the data
   - Normalizes numeric features
   - Encodes categorical variables
   - Exports training data to Parquet format

2. **Model Training** (`_02_binomial_classifier.py`):
   - Trains an XGBoost binary classifier
   - Uses grid search for hyperparameter optimization
   - Evaluates model performance
   - Logs model metrics and parameters to MLflow
   - Exports prediction results

3. **Analysis** (`binomial_classifier_analysis.ipynb`):
   - Visualizes model results
   - Identifies false positives and false negatives
   - Analyzes feature correlations

The `_00_error_correction.py` script is available for ad hoc error correction when needed.

## Integration

This project will eventually be converted into a microservice to work alongside [automatic-transmission](https://github.com/x81k25/automatic-transmission) as an ML component in the pipeline for downloading media.

## MLflow Tracking

This project uses MLflow to track model experiments. For information on setting up and using MLflow, refer to the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).

## License

MIT License