"""Sentiment Analysis Service"""

from app.ml.phobert.predictor import PhobertPredictor
from app.schemas.feedback import SentimentLabel
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound PhoBERT inference
executor = ThreadPoolExecutor(max_workers=4)


class SentimentService:
    """Service for sentiment analysis operations"""
    
    def __init__(self):
        self.predictor = PhobertPredictor.get_instance()
    
    async def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of Vietnamese text asynchronously
        
        Args:
            text: Vietnamese text to analyze
        
        Returns:
            Dict with sentiment label and score
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be non-empty string")
        
        # Run CPU-intensive model in thread pool
        loop = asyncio.get_event_loop()
        sentiment, score = await loop.run_in_executor(
            executor,
            self.predictor.predict,
            text
        )
        
        return {
            "sentiment": sentiment,
            "score": float(score)
        }
    
    async def analyze_batch_sentiments(self, texts: list[str]) -> list[dict]:
        """
        Analyze sentiments for multiple texts
        
        Args:
            texts: List of Vietnamese texts
        
        Returns:
            List of sentiment analysis results
        """
        tasks = [self.analyze_sentiment(text) for text in texts]
        return await asyncio.gather(*tasks)
