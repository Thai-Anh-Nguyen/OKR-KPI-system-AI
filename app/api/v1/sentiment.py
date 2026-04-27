"""Sentiment Analysis API Endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from app.schemas.feedback import SentimentResponse, FeedbackSentimentRequest
from app.services.sentiment_svc import SentimentService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_sentiment_service():
    """Dependency injection for sentiment service"""
    return SentimentService()


@router.post("/", response_model=SentimentResponse)
async def analyze_sentiment(
    request: FeedbackSentimentRequest,
    service: SentimentService = Depends(get_sentiment_service)
):
    """
    Analyze sentiment of Vietnamese text
    
    - **text**: Vietnamese feedback text to analyze
    
    Returns sentiment classification (POSITIVE/NEGATIVE/NEUTRAL/MIXED) and confidence score
    """
    try:
        result = await service.analyze_sentiment(request.text)
        return SentimentResponse(**result)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def sentiment_health():
    """Health check for sentiment service"""
    return {"status": "ok", "service": "sentiment-analysis"}
