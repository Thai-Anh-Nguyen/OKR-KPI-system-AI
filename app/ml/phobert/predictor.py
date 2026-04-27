"""PhoBERT Sentiment Analysis Model"""

from typing import Literal
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except ImportError:
    pipeline = None
    logger.warning("Transformers not available, using rule-based fallback")

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
            if pipeline is None:
                logger.warning("Transformers pipeline not available, using rule-based fallback.")
                self._model = None
            else:
                try:
                    self._model = pipeline(
                        "text-classification",
                        model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
                    )
                    logger.info("PhoBERT model loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load PhoBERT model: {e}. Using rule-based fallback.")
                    self._model = None  # Explicitly set to None for fallback mode
    
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
        
        # Use ML model if available, otherwise fallback to rule-based
        if self._model:
            try:
                result = self._model(text)[0]
                label = result['label'].lower()
                score = result['score']
                
                # Map model labels to sentiment labels
                if 'LABEL_2' in label or 'POSITIVE' in label.upper():
                    sentiment = 'POSITIVE'
                elif 'LABEL_1' in label or 'NEUTRAL' in label.upper():
                    sentiment = 'NEUTRAL'
                else:
                    sentiment = 'NEGATIVE'
                
                logger.info(f"ML model prediction: {sentiment} (score: {score:.4f})")
                return sentiment, score
            except Exception as e:
                logger.warning(f"ML model prediction failed: {e}. Using rule-based fallback.")
                return self._rule_based_sentiment(text)
        else:
            logger.info("Using rule-based sentiment analysis")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> tuple[SentimentLabel, float]:
        """
        Rule-based sentiment analysis fallback
        """
        text_lower = text.lower()
        
        positive_words = ['tốt', 'tuyệt vời', 'hài lòng', 'hỗ trợ', 'tích cực', 'vui', 'yêu thích', 'xuất sắc']
        negative_words = ['xấu', 'tồi tệ', 'không hài lòng', 'áp lực', 'khó khăn', 'buồn', 'thất bại', 'tệ']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'POSITIVE', 0.7
        elif negative_count > positive_count:
            return 'NEGATIVE', 0.7
        else:
            return 'NEUTRAL', 0.5
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
