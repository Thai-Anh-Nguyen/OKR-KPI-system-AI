import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Global variable to hold the pipeline in memory
_sentiment_pipeline = None

def load_phobert_model():
    """
    Loads the fine-tuned PhoBERT model into memory.
    This should be called exactly once during FastAPI startup.
    """
    global _sentiment_pipeline
    logger.info("Loading PhoBERT sentiment model (wonrax/phobert-base-vietnamese-sentiment)...")
    
    # We use a popular open-source fine-tuned model for Vietnamese sentiment
    # It outputs POS (Positive), NEG (Negative), and NEU (Neutral)
    _sentiment_pipeline = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment")
    
    logger.info("PhoBERT sentiment model loaded successfully.")

def get_sentiment_pipeline():
    """
    Returns the loaded pipeline.
    """
    if _sentiment_pipeline is None:
        raise RuntimeError("PhoBERT model is not loaded. Call load_phobert_model() first.")
    return _sentiment_pipeline
