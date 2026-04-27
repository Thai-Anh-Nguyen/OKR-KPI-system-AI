import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.sentiment import router as sentiment_router
from app.api.risk import router as risk_router
from app.api.rag import router as rag_router
from app.ml.phobert.loader import load_phobert_model
from app.ml.knn.predictor import load_knn_model

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the heavy ML models into memory
    logging.info("Starting up FastAPI application...")
    load_phobert_model()
    load_knn_model()
    yield
    # Shutdown: Clean up resources if necessary
    logging.info("Shutting down FastAPI application...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Include Routers
app.include_router(sentiment_router, prefix=settings.API_V1_STR, tags=["sentiment"])
app.include_router(risk_router, prefix=settings.API_V1_STR, tags=["risk"])
app.include_router(rag_router, prefix=settings.API_V1_STR, tags=["rag"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
