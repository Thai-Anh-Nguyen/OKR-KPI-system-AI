"""PhoBERT Sentiment Analysis Model"""

from typing import Literal
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

SentimentLabel = Literal["POSITIVE", "NEUTRAL", "NEGATIVE", "MIXED"]


class PhobertPredictor:
    """Wrapper for PhoBERT sentiment classification"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            logger.info("Loading PhoBERT model for sentiment analysis...")
            try:
                self._model = pipeline(
                    "text-classification",
                    model="nlptown/bert-base-multilingual-uncased-sentiment"
                )
                logger.info("PhoBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PhoBERT model: {e}")
                raise
    
    def predict(self, text: str) -> tuple[SentimentLabel, float]:
        """
        Classify sentiment of Vietnamese text
        
        Args:
            text: Vietnamese text to classify
            
        Returns:
            Tuple of (sentiment_label, confidence_score)
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        
        try:
            results = self._model(text, truncation=True, max_length=512)
            
            if not results:
                logger.warning(f"No results from model for text: {text[:50]}")
                return "NEUTRAL", 0.5
            
            result = results[0]
            label = result['label'].upper()
            score = result['score']
            
            # Map multilingual model labels to our schema
            label_mapping = {
                '5 STARS': 'POSITIVE',
                '4 STARS': 'POSITIVE',
                '3 STARS': 'NEUTRAL',
                '2 STARS': 'NEGATIVE',
                '1 STAR': 'NEGATIVE',
                'POSITIVE': 'POSITIVE',
                'NEGATIVE': 'NEGATIVE',
                'NEUTRAL': 'NEUTRAL',
            }
            
            sentiment = label_mapping.get(label, 'NEUTRAL')
            return sentiment, score
            
        except Exception as e:
            logger.error(f"Error during sentiment prediction: {e}")
            raise
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
