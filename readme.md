# reel-driver

personal media curation algorithm trained on my personally labeled data

## overview

Ever opened app after app on you SmartTV and you were greeted by a top of row of squares of content you mostly weren't interested in? Well, if so, then this may be the repo you've been looking for. I've been repeatedly disappointed by content curation on the big streamers. Either they promote content I'm not interested in, I have to dig for content I do want, or it may display content to me only to discover after clicking on it that I need another subscription or a purchase to access it.

The intention of this model is to create a model that ingests personalize training data in order to create a model that can run inferences on new media items and tell you whether or not you'd be into it! The data samples I have in the `/data` folder contain the training data and analysis results based off of my own preferences.   

Currently I am working on tuning the model for local development in order to prove the model's viability. Eventually I will integrate this model into my media microservice ecosystem and run it on my locally deployed k3s cluster. When the model prototype is done, I am going to flesh out this service as a FaspAPI accessible microservice. 

Currently the datasets I am using filter for only movies, but it theoretically should be able to handle other media types as well, perhaps with same slight modifications based on what metadata is avialable for training of that media type. 

## project structure

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

## setup

### prerequisites

- Python 3.12+
- PostgreSQL database with appropriate schema setup
- MLflow server (for model tracking)

### environment variables

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

### installation

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

# regenerate requirements.txt upon requirements alteration
uv pip compile requirements.in -o requirements.txt
```

## modeling

### problem formulation 

For the initial version of this model I am going to go with a binomial classifier, largely because that is the type of model that will best fit my current model training data. As an output of the services deployed via my [automatic-transmission](https://github.com/x81k25/automatic-transmission) repository, I have multiple status flags attached to several thousand media records that can be used a a true/false label for media selection. At the end of the day, I am making a binary decision, whether or not to include the model in my media library.

It would however be interesting to potentially try a multi-class classification problem. When discussing with my wife, we have considered labeling the data as one of these possibilities `["would-not-watch", "would watch", "woud watch multiple times"]` or somthing analogous. This likely would increase decrease the probability of getting false negatives on movies that we would enjoy the most, by giving them thier own distinct class. A multi-class classifier would also be fun, because while a majority of the purpose of this project is to build a microservice that will go into the rest of my media ecosystem, it is also really fun to try and algorithmically break down my own preferences; and a multi-class classifier would allow me to do so in an even more in-depth way.

Proceeding down that same trail, a regression model could work here as well. That would require even more of a concerted effor to label the data. I am currently sitting at over 3k movies, so each of them would need to be manually numerically labeled in order to capture my prefences. As a person that is acutely aware of the inherent flaws of human-decision making, it would pain me somewhat to utilize a continuous numeric label that was entirely subjective. 

### algorithm selection



### evaluation criteria



## workflow

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

## integration

This project will eventually be converted into a microservice to work alongside [automatic-transmission](https://github.com/x81k25/automatic-transmission) as an ML component in the pipeline for downloading media.

## MLflow tracking

This project uses MLflow to track model experiments. For information on setting up and using MLflow, refer to the [official MLflow documentation](https://mlflow.org/docs/latest/index.html).


## FAST-API service

### Testing the API Locally

To test the Reel Driver API locally before containerization:

1. Ensure Install required dependencies:

```bash
pip install fastapi uvicorn
```

2. Start the API server:
```bash
uvicorn app.main:app --reload
```

3. Test the endpoints:

**Health check:**

```bash
curl http://localhost:8000/health
```
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

**Single prediction:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hash": "test123",
    "release_year": 2010,
    "genre": ["Drama", "Comedy"],
    "language": ["en"],
    "metascore": 75,
    "rt_score": 85,
    "imdb_rating": 7.5,
    "imdb_votes": 10000
  }'
```
```powershell
$body = @{
  hash = "test123"
  release_year = 2010
  genre = @("Drama", "Comedy")
  language = @("en")
  metascore = 75
  rt_score = 85
  imdb_rating = 7.5
  imdb_votes = 10000
}
$json = ConvertTo-Json $body
Invoke-RestMethod -Uri http://localhost:8000/api/predict -Method Post -Body $json -ContentType "application/json"
```

**Batch prediction:**

```bash
curl -X POST http://localhost:8000/api/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "hash": "test123",
        "release_year": 2010,
        "genre": ["Drama", "Comedy"],
        "language": ["en"],
        "metascore": 75,
        "rt_score": 85,
        "imdb_rating": 7.5,
        "imdb_votes": 10000
      },
      {
        "hash": "test456",
        "release_year": 2020,
        "genre": ["Horror", "Thriller"],
        "language": ["en"],
        "metascore": 45,
        "rt_score": 30,
        "imdb_rating": 4.8,
        "imdb_votes": 5000
      }
    ]
  }'
```
```powershell
$item1 = @{
  hash = "test123"
  release_year = 2010
  genre = @("Drama", "Comedy")
  language = @("en")
  metascore = 75
  rt_score = 85
  imdb_rating = 7.5
  imdb_votes = 10000
}

$item2 = @{
  hash = "test456"
  release_year = 2020
  genre = @("Horror", "Thriller")
  language = @("en")
  metascore = 45
  rt_score = 30
  imdb_rating = 4.8
  imdb_votes = 5000
}

$batch = @{
  items = @($item1, $item2)
}

$json = ConvertTo-Json $batch -Depth 3
Invoke-RestMethod -Uri http://localhost:8000/api/predict_batch -Method Post -Body $json -ContentType "application/json"
```

4. Access the interactive API documentation at http://localhost:8000/docs

5. To stop the server when finished:
```bash
# Press Ctrl+C in the terminal running the server
```

6. Kill any running instances that weren't terminated properly
```bash
pkill -9 python
```
```powershell
Get-Process -Name python | Stop-Process -Force
```

### Building and running in docker locally

**budilng image**
```bash
# Regular build
docker build -t reel-driver-image -f app/dockerfile .

# Force rebuild without cache
docker build --no-cache -t reel-driver-image -f app/dockerfile .
```

**running the container**
```bash
# Run container in foreground
docker run -p 8000:8000 --name reel-driver-container --env-file app/.env reel-driver-image

# Run container in background
docker run -d -p 8000:8000 --name reel-driver-container --env-file app/.env reel-driver-image

# Stop the container
docker stop reel-driver-container

# remove container
sudo docker rm reel-driver-container

# delete image
docker rmi reel-driver-image

# delete all images
docker rmi $(docker images -q) -f
```

**with docker compose**
```bash
# Build and start services
docker compose -f app/docker-compose.yaml up

# Build with no cache and start
docker compose -f app/docker-compose.yaml build --no-cache
docker compose -f app/docker-compose.yaml up

# Run in background
docker compose -f app/docker-compose.yaml up -d

# Stop services
docker compose -f app/docker-compose.yaml down
```

**troubleshooting**
```bash
# View logs
docker logs reel-driver-container

# Shell into container
docker exec -it reel-driver-container bash

# Check container status
docker ps -a | grep reel-driver-container
```


## license

MIT License

