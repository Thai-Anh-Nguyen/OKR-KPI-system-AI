# OKR-KPI System AI Microservice - Setup Guide

This document provides complete setup and deployment instructions for the AI microservice.

## Project Structure

```
app/
├── api/v1/              # API routes
│   ├── sentiment.py     # Vietnamese sentiment analysis endpoints
│   ├── risk.py          # Risk prediction endpoints
│   └── rag.py           # Alert generation endpoints
├── core/                # Core configurations
│   ├── config.py        # Settings management
│   └── logger.py        # Logging setup
├── ml/                  # Machine Learning models
│   ├── phobert/         # PhoBERT sentiment model
│   │   ├── loader.py
│   │   └── predictor.py
│   └── knn/             # KNN risk classification model
│       ├── trainer.py
│       └── predictor.py
├── schemas/             # Pydantic data models
│   ├── feedback.py      # Sentiment request/response
│   ├── employee.py      # Risk analysis schemas
│   └── rag.py           # Alert generation schemas
├── services/            # Business logic layer
│   ├── sentiment_svc.py # Sentiment service
│   ├── risk_svc.py      # Risk analysis service
│   └── rag_svc.py       # Alert generation service
└── __init__.py

main.py                 # FastAPI application entry point
requirements.txt        # Python dependencies
.env                    # Environment configuration
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `PHOBERT_MODEL_NAME`: Model for sentiment analysis
- `KNN_MODEL_DIR`: Directory for KNN model artifacts
- `LLM_API_KEY`: API key for LLM provider (optional)

## Running the Service

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Sentiment Analysis

**POST** `/api/v1/sentiment/`

Analyze Vietnamese text sentiment

Request:
```json
{
  "text": "Công ty là nơi làm việc tuyệt vời"
}
```

Response:
```json
{
  "sentiment": "POSITIVE",
  "score": 0.95
}
```

### 2. Risk Analysis

**POST** `/api/v1/risk-analysis/`

Predict employee risk level

Request:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "features": {
    "kpi_completion_rate": 0.85,
    "checkin_delay_days": 5.2,
    "feedback_sentiment_score": 0.3,
    "objective_participation_ratio": 1.2
  }
}
```

Response:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "risk_label": "medium",
  "risk_score": 0.5,
  "features": { ... }
}
```

**POST** `/api/v1/risk-analysis/batch`

Batch risk analysis for multiple employees

### 3. Alert Generation

**POST** `/api/v1/generate-alert/`

Generate HR alert for high-risk employee

Request:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "risk_score": 0.75,
  "risk_label": "high",
  "context": {
    "features": { ... }
  }
}
```

Response:
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "alert_id": "alert-xxx",
  "content": {
    "title": "⚠️ High Risk Alert",
    "description": "Employee showing critical performance indicators",
    "priority": "high",
    "recommendations": [ ... ]
  },
  "generated_at": "2024-04-27T10:30:00Z"
}
```

## Testing

### Run Test Notebook

Open `Test_OKR_KPI_AI_Model.ipynb` in Jupyter:

```bash
jupyter notebook Test_OKR_KPI_AI_Model.ipynb
```

The notebook includes:
- Sentiment analysis test cases
- Risk prediction tests
- Alert generation demonstrations
- Integration tests
- Performance benchmarks

### Run Unit Tests

```bash
pytest tests/ -v
```

## Model Training

### KNN Risk Model

Train a new KNN model from data:

```bash
python train_knn_risk_model.py --data kpi_okr_data.xlsx --output ./models
```

### Prediction

Make predictions using trained model:

```bash
python predict_risk.py \
  --kpi 0.85 \
  --delay 5.2 \
  --sentiment 0.3 \
  --obj-ratio 1.2 \
  --model-dir ./models
```

## Performance Considerations

### Model Loading

- PhoBERT model loads on application startup (lifespan event)
- KNN model loads on first request with fallback to rule-based prediction
- Models are cached as singletons to avoid reloading

### Async Operations

- Sentiment analysis runs in ThreadPoolExecutor (CPU-bound)
- Risk prediction runs synchronously (already optimized)
- Batch operations support concurrent processing

### Monitoring

Check service health:

```bash
curl http://localhost:8000/health
```

View interactive API documentation:

```
http://localhost:8000/docs
```

## Integration with Node.js Backend

### Sentiment Analysis Flow

1. Node.js feedback service → POST `/api/v1/sentiment/`
2. Service analyzes text and returns sentiment
3. Node.js stores result in database

### Risk Analysis Flow

1. Node.js ETL job → POST `/api/v1/risk-analysis/batch`
2. Service predicts risk for all employees
3. Results used to trigger alerts if risk > 0.5

### Alert Generation Flow

1. Risk score triggers RAG pipeline
2. Service retrieves HR policies
3. Generates actionable alert
4. Node.js stores alert in database

## Troubleshooting

### Model Download Issues

If models fail to download:

```bash
# Manually download PhoBERT
python -c "from transformers import pipeline; pipeline('text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment')"
```

### Memory Issues

For production with limited memory:

1. Use quantized models
2. Reduce batch size: `MAX_BATCH_SIZE=50`
3. Increase workers gradually

### Performance Optimization

1. Use GPU if available (set `CUDA_VISIBLE_DEVICES`)
2. Increase thread pool: `THREAD_POOL_WORKERS=8`
3. Enable response caching
4. Use load balancer for horizontal scaling

## Production Deployment

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t okr-kpi-ai .
docker run -p 8000:8000 okr-kpi-ai
```

### Environment Variables

Set in production:

```bash
export LOG_LEVEL=INFO
export WORKERS=4
export PORT=8000
```

## Support and Documentation

- API Docs: http://localhost:8000/docs
- Test Cases: `Test_OKR_KPI_AI_Model.ipynb`
- Architecture: See `AGENT_CONTEXT.md`
- Code: Well-commented with type hints

## License

Proprietary - OKR-KPI System
