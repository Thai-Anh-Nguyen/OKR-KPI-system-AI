# OKR KPI System - AI Microservice

This repository contains the Python FastAPI microservice that powers the AI capabilities of the OKR KPI System. It works alongside the main Node.js backend to provide advanced machine learning and generative AI features.

## Responsibilities

This microservice handles heavy computational and AI tasks that are not suitable for Node.js:
1. **PhoBERT Sentiment Analysis**: Classifies Vietnamese employee feedback into POSITIVE, NEUTRAL, NEGATIVE, or MIXED.
2. **KNN Risk Clustering**: Calculates Euclidean distances between employee feature vectors to assign peer-group risk labels (Low/Medium/High).
3. **RAG Pipeline**: Retrieves company policies from a vector database (pgvector) and uses an LLM to generate actionable, human-readable HR alerts.

## Architecture

*   **Framework**: FastAPI (Python)
*   **NLP Model**: `vinai/phobert-base-v2` (via Hugging Face `transformers`)
*   **Machine Learning**: `scikit-learn` (for K-Nearest Neighbors)
*   **Vector Database**: PostgreSQL with `pgvector` extension
*   **LLM Orchestration**: LangChain / LlamaIndex

## Interaction with Node.js Backend

This service acts as a RESTful API worker:
*   The Node.js backend calls `POST /api/v1/sentiment` when a new feedback is submitted.
*   The Node.js ETL Cron Job calls `POST /api/v1/risk-analysis` after calculating statistical baselines.
*   If risk scores exceed thresholds, this service triggers the RAG pipeline and updates the database.

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies (once `requirements.txt` is populated):
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```
