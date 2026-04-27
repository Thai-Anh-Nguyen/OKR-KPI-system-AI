from pydantic import BaseModel, Field
from typing import Optional, Literal


class FeedbackBase(BaseModel):
    text: str = Field(..., description="Vietnamese feedback text")
    user_id: Optional[str] = None
    created_at: Optional[str] = None


class FeedbackSentimentRequest(BaseModel):
    text: str = Field(..., description="Vietnamese feedback text to analyze")


SentimentLabel = Literal["POSITIVE", "NEUTRAL", "NEGATIVE", "MIXED"]


class SentimentResponse(BaseModel):
    sentiment: SentimentLabel
    score: float = Field(..., description="Confidence score (0.0-1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sentiment": "POSITIVE",
                "score": 0.95
            }
        }


class FeedbackAnalysisResponse(FeedbackBase):
    sentiment: Optional[SentimentLabel] = None
    sentiment_score: Optional[float] = None
