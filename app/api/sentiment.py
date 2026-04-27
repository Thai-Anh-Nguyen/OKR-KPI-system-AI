from fastapi import APIRouter, HTTPException
from app.models.schemas import SentimentRequest, SentimentResponse
import app.services.phobert_svc as sentiment_service

router = APIRouter()

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyzes the sentiment of a given text (Vietnamese).
    Uses the loaded PhoBERT model in the background.
    """
    try:
        result = await sentiment_service.analyze_sentiment_async(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
