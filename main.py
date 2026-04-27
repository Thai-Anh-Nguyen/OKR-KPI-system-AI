from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from app.api.v1 import sentiment, risk, rag
from app.core.config import settings
from app.core.logger import logger
from app.ml.phobert.predictor import PhobertPredictor
from app.ml.knn.predictor import KNNRiskPredictor


# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifecycle
    Load ML models at startup, cleanup at shutdown
    """
    logger.info("=== Application Startup ===")
    
    # Load ML models at startup
    try:
        logger.info("Initializing PhoBERT model...")
        PhobertPredictor.get_instance()
        logger.info("✓ PhoBERT model loaded")
    except Exception as e:
        logger.warning(f"⚠ PhoBERT model failed to load: {e}. Using fallback.")
    
    try:
        logger.info("Initializing KNN Risk model...")
        KNNRiskPredictor.get_instance(settings.KNN_MODEL_DIR)
        logger.info("✓ KNN Risk model loaded")
    except Exception as e:
        logger.warning(f"⚠ KNN Risk model failed to load: {e}. Using rule-based fallback.")
    
    logger.info("=== Application Ready ===\n")
    
    yield
    
    # Cleanup at shutdown
    logger.info("\n=== Application Shutdown ===")
    logger.info("Cleaning up resources...")
    logger.info("✓ Application shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI endpoints for sentiment analysis, risk prediction, and RAG",
    version=settings.VERSION,
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.debug(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.3f}s"
    )
    return response


# Include routers
app.include_router(
    sentiment.router,
    prefix="/api/v1/sentiment",
    tags=["Sentiment Analysis"]
)
app.include_router(
    risk.router,
    prefix="/api/v1/risk-analysis",
    tags=["Risk Analysis"]
)
app.include_router(
    rag.router,
    prefix="/api/v1/generate-alert",
    tags=["RAG Alert Generation"]
)


@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to OKR-KPI System AI Microservice",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
