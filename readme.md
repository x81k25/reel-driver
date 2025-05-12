# reel-driver

A personal media curation algorithm trained on personally labeled data.

## Overview

Ever opened app after app on your SmartTV and you were greeted by a top row of squares of content you mostly weren't interested in? Well, if so, then this may be the repo you've been looking for. I've been repeatedly disappointed by content curation on the big streamers. Either they promote content I'm not interested in, I have to dig for content I do want, or it may display content to me only to discover after clicking on it that I need another subscription or a purchase to access it.

The intention of this model is to create a model that ingests personalize training data in order to create a model that can run inferences on new media items and tell you whether or not you'd be into it! The data samples I have in the `/data` folder contain the training data and analysis results based off of my own preferences.   

The project is structured into three main layers:

1. **Project Level** - The overarching configuration and coordination of all components
2. **Training Level** - The code for data processing and model training
3. **API Level** - The FastAPI service for serving model predictions

## 1. Project Level

The project level coordinates the overall system and provides the structure for the entire application.

### Project Structure

```
reel-driver/
├── app/                    # API service code (see API Level)
├── data/                   # Data files (not included in repo due to size)
│   ├── media.parquet
│   ├── binomial_classifier_training_data.parquet
│   ├── binomial_classifier_results.parquet
│   ├── false_positives.json
│   └── false_negatives.json
├── model_artifacts/        # Trained model files
│   ├── normalization.json  # Normalization parameters
│   └── xgb_model.json      # Trained XGBoost model
├── notebooks/              # Jupyter notebooks for analysis
│   └── binomial_classifier_analysis.ipynb
├── src/                    # Training code (see Training Level)
├── training.py             # Main script to run training pipeline
├── predict.py              # Test script for inference
├── .gitignore
├── README.md
├── requirements.in         # Input file for dependency management
└── requirements.txt        # Generated dependencies with pinned versions
```

### Prerequisites

For the complete project:
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

# Regenerate requirements.txt upon requirements alteration
uv pip compile requirements.in -o requirements.txt
```

### Project Workflow

Currently I am working on tuning the model for local development in order to prove the model's viability. Eventually I will integrate this model into my media microservice ecosystem and run it on my locally deployed k3s cluster. When the model prototype is done, I am going to flesh out this service as a FastAPI accessible microservice.

Currently the datasets I am using filter for only movies, but it theoretically should be able to handle other media types as well, perhaps with some slight modifications based on what metadata is available for training of that media type.

The project follows this workflow:

1. **Data Extraction**: Extract data from PostgreSQL database
2. **Data Preparation**: Transform and clean data, normalize features, encode categorical variables
3. **Model Training**: Train an XGBoost binary classifier with optimized hyperparameters
4. **API Deployment**: Deploy the model as a FastAPI service

Run the entire training pipeline:

```bash
python training.py
```

