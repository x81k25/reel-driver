## 3. API Level

The API level provides a FastAPI service for serving model predictions.

### API Structure

The API code is organized in the `app/` directory:

```
app/
├── core/                  # Core application components
│   └── config.py          # Application settings
├── models/                # Data models
│   ├── __init__.py
│   ├── api.py             # API request/response models
│   └── media_prediction_input.py  # Input validation model
├── routers/               # API route handlers
│   └── prediction.py      # Prediction endpoints
├── services/              # Business logic
│   └── predictor.py       # XGBoost predictor service
├── dockerfile.api         # Docker configuration
├── docker-compose.yaml    # Docker Compose configuration
├── main.py                # FastAPI application entry point
├── requirements.in        # API-specific dependencies
└── requirements.txt       # Generated API dependencies
```

### API Endpoints

The API exposes the following endpoints:

- `GET /health` - Health check endpoint
- `GET /` - API root with documentation links
- `POST /api/predict` - Single prediction endpoint
- `POST /api/predict_batch` - Batch prediction endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

### Testing the API Locally

To test the Reel Driver API locally before containerization:

1. Ensure required dependencies are installed:

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
curl http://localhost:8000/reel-driver/health
```
```powershell
Invoke-RestMethod -Uri http://localhost:8000/reel-driver/health
```

*Sample Response:*
```json
{
  "status": "healthy",
  "model_features": 245,
  "normalization_fields": 10
}
```

**Single prediction:**

```bash
curl -X POST http://localhost:8000/reel-driver/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "imdb_id": "tt0111161",
    "release_year": 1994,
    "genre": ["Drama", "Crime"],
    "spoken_languages": ["en"],
    "original_language": "en",
    "origin_country": ["US"],
    "production_countries": ["US"],
    "production_status": "Released",
    "metascore": 82,
    "rt_score": 91,
    "imdb_rating": 92.0,
    "imdb_votes": 2800000,
    "tmdb_rating": 8.7,
    "tmdb_votes": 26000,
    "budget": 25000000,
    "revenue": 16000000,
    "runtime": 142,
    "tagline": "Fear can hold you prisoner. Hope can set you free.",
    "overview": "Framed in the 1940s for the double murder of his wife and her lover..."
  }'
```
```powershell
$body = @{
  imdb_id = "tt0111161"
  release_year = 1994
  genre = @("Drama", "Crime")
  spoken_languages = @("en")
  original_language = "en"
  origin_country = @("US")
  production_countries = @("US")
  production_status = "Released"
  metascore = 82
  rt_score = 91
  imdb_rating = 92.0
  imdb_votes = 2800000
  tmdb_rating = 8.7
  tmdb_votes = 26000
  budget = 25000000
  revenue = 16000000
  runtime = 142
  tagline = "Fear can hold you prisoner. Hope can set you free."
  overview = "Framed in the 1940s for the double murder of his wife and her lover..."
}
$json = ConvertTo-Json $body
Invoke-RestMethod -Uri http://localhost:8000/reel-driver/api/predict -Method Post -Body $json -ContentType "application/json"
```

*Sample Response:*
```json
{
  "imdb_id": "tt0111161",
  "prediction": true,
  "probability": 0.89
}
```

**Batch prediction:**

```bash
curl -X POST http://localhost:8000/reel-driver/api/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "imdb_id": "tt0111161",
        "release_year": 1994,
        "genre": ["Drama", "Crime"],
        "spoken_languages": ["en"],
        "original_language": "en",
        "origin_country": ["US"],
        "production_countries": ["US"],
        "production_status": "Released",
        "metascore": 82,
        "rt_score": 91,
        "imdb_rating": 92.0,
        "imdb_votes": 2800000,
        "tmdb_rating": 8.7,
        "tmdb_votes": 26000,
        "budget": 25000000,
        "revenue": 16000000,
        "runtime": 142,
      },
      {
        "imdb_id": "tt0468569",
        "release_year": 2008,
        "genre": ["Action", "Crime", "Drama"],
        "spoken_languages": ["en"],
        "original_language": "en",
        "origin_country": ["US"],
        "production_countries": ["US"],
        "production_status": "Released",
        "metascore": 84,
        "rt_score": 94,
        "imdb_rating": 90.0,
        "imdb_votes": 2700000,
        "tmdb_rating": 8.5,
        "tmdb_votes": 32000,
        "budget": 185000000,
        "revenue": 1005000000,
        "runtime": 152,
      }
    ]
  }'
```
```powershell
$item1 = @{
  imdb_id = "tt0111161"
  release_year = 1994
  genre = @("Drama", "Crime")
  spoken_languages = @("en")
  original_language = "en"
  origin_country = @("US")
  production_countries = @("US")
  production_status = "Released"
  metascore = 82
  rt_score = 91
  imdb_rating = 92.0
  imdb_votes = 2800000
  tmdb_rating = 8.7
  tmdb_votes = 26000
  budget = 25000000
  revenue = 16000000
  runtime = 142
}

$item2 = @{
  imdb_id = "tt0468569"
  release_year = 2008
  genre = @("Action", "Crime", "Drama")
  spoken_languages = @("en")
  original_language = "en"
  origin_country = @("US")
  production_countries = @("US")
  production_status = "Released"
  metascore = 84
  rt_score = 94
  imdb_rating = 90.0
  imdb_votes = 2700000
  tmdb_rating = 8.5
  tmdb_votes = 32000
  budget = 185000000
  revenue = 1005000000
  runtime = 152
}

$batch = @{
  items = @($item1, $item2)
}

$json = ConvertTo-Json $batch -Depth 3
Invoke-RestMethod -Uri http://localhost:8000/reel-driver/api/predict_batch -Method Post -Body $json -ContentType "application/json"
```

*Sample Response:*
```json
{
  "results": [
    {
      "imdb_id": "tt0111161",
      "prediction": true,
      "probability": 0.89
    },
    {
      "imdb_id": "tt0468569",
      "prediction": true,
      "probability": 0.95
    }
  ]
}
```

4. Access the interactive API documentation at http://localhost:8000/reel-driver/docs

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

### Building and Running in Docker Locally

**Building the image**
```bash
# Regular build
docker build -t reel-driver-image -f app/dockerfile .

# Force rebuild without cache
docker build --no-cache -t reel-driver-image -f app/dockerfile .
```

**Running the container**
```bash
# Run container in foreground
docker run -p 8000:8000 --name reel-driver-container --env-file app/.env reel-driver-image

# Run container in background
docker run -d -p 8000:8000 --name reel-driver-container --env-file app/.env reel-driver-image
```

**With Docker Compose**
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

**container clean-up**
```bash
# stop container
docker stop reel-driver-container

# remove container
docker rm reel-driver-container

# remove image
docker image rmi reel-driver-image

# full clean-up
docker stop reel-driver-container && docker rm reel-driver-container && docker image rmi reel-driver-image
```

**Troubleshooting**
```bash
# View logs
docker logs reel-driver-container

# Shell into container
docker exec -it reel-driver-container bash

# Check container status
docker ps -a | grep reel-driver-container
```

## Integration

This project will eventually be converted into a microservice to work alongside [automatic-transmission](https://github.com/x81k25/automatic-transmission) as an ML component in the pipeline for downloading media.

## License

MIT License