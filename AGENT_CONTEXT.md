# Agent Context — FastAPI AI Microservice

## 1. Project Purpose
You are assisting in building a Python **FastAPI** microservice. This service is a companion to a Node.js/Prisma backend. 
The main Node.js backend handles CRUD operations, Cron Jobs (ETL), and basic statistical math.
This FastAPI service handles:
- Vietnamese NLP (using PhoBERT)
- Machine Learning Clustering (using Scikit-Learn KNN)
- Retrieval-Augmented Generation (RAG)

## 2. Target Folder Structure
```text
.
├── app/
│   ├── api/
│   │   ├── sentiment.py  # Endpoints for PhoBERT sentiment
│   │   ├── risk.py       # Endpoints for KNN clustering
│   │   └── rag.py        # Endpoints for generating LLM alerts
│   ├── core/
│   │   └── config.py     # Environment variables
│   ├── models/           # Pydantic schemas for request/response
│   ├── services/         # Business logic gluing API to ML models
│   ├── ml/
│   │   ├── phobert/
│   │   └── knn/
│   └── rag/
├── main.py
├── requirements.txt
└── .env
```

## 3. Integration Points with Node.js
Do NOT attempt to write database migration scripts or CRUD logic here unless it specifically relates to AI. The Node.js app manages the PostgreSQL database schema via Prisma.
When this Python service needs data, it either:
1. Receives it directly in the HTTP Request Payload from Node.js.
2. Connects to PostgreSQL directly (Read-only for features, Read/Write for updating `AIAlerts` or `RiskScores`).

### Endpoint 1: Sentiment Analysis
- **Called by**: Node.js `feedback.service.js` upon feedback creation.
- **Input**: `{ "text": "Vietnamese string" }`
- **Logic**: Use Hugging Face `transformers` with a PhoBERT model to classify sentiment.
- **Output**: `{ "sentiment": "POSITIVE|NEGATIVE|NEUTRAL|MIXED", "score": float }`

### Endpoint 2: Risk Clustering
- **Called by**: Node.js ETL Job (`behaviorAnalysis.job.js`) after it computes baseline statistics.
- **Input**: `{ "user_id": uuid, "features": [...] }`
- **Logic**: Use `sklearn.neighbors.KNeighborsClassifier` to find peer groups and assign a `knn_risk_label`. Calculate the final `risk_score`.
- **Output**: `{ "knn_risk_label": "LOW|MEDIUM|HIGH", "risk_score": float }`

### Endpoint 3: RAG Alert Generation
- **Trigger**: Called automatically if `risk_score >= 0.5`.
- **Logic**: Retrieve HR policies from `pgvector`. Send context + employee metrics to an LLM (Claude/OpenAI). Generate JSON alert.
- **Output**: JSON conforming to the `AIAlerts` schema.

## 4. Coding Rules for AI Agents
- Always use **FastAPI** best practices (Dependency Injection, APIRouters, Pydantic v2).
- Ensure ML models are loaded into memory globally on application startup (lifespan events), NOT re-loaded on every request.
- Use asynchronous programming (`async def`) for API endpoints, but run heavy ML inference (like PhoBERT CPU processing) in a `ThreadPoolExecutor` to avoid blocking the event loop.
- All code must be strongly typed using Python type hints.
