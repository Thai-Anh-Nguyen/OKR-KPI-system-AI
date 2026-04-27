from fastapi import FastAPI
from app.api.v1 import sentiment, risk, rag

app = FastAPI(
    title="OKR-KPI System AI Microservice",
    description="AI endpoints for sentiment analysis, risk prediction, and RAG",
    version="1.0.0"
)

# Include routers
app.include_router(sentiment.router, prefix="/api/v1/sentiment", tags=["Sentiment"])
app.include_router(risk.router, prefix="/api/v1/risk-analysis", tags=["Risk Analysis"])
app.include_router(rag.router, prefix="/api/v1/generate-alert", tags=["RAG Alert"])

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "okr-kpi-system-ai"}
