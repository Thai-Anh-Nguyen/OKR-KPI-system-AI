"""PhoBERT model loader"""

from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class PhobertModelLoader:
    """Utility to load PhoBERT models and tokenizers"""
    
    MODEL_NAME = "vinai/phobert-base-v2"
    
    @staticmethod
    def load_model():
        """Load PhoBERT model"""
        logger.info(f"Loading model from {PhobertModelLoader.MODEL_NAME}")
        model = AutoModel.from_pretrained(PhobertModelLoader.MODEL_NAME)
        logger.info("Model loaded successfully")
        return model
    
    @staticmethod
    def load_tokenizer():
        """Load PhoBERT tokenizer"""
        logger.info(f"Loading tokenizer from {PhobertModelLoader.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(PhobertModelLoader.MODEL_NAME)
        logger.info("Tokenizer loaded successfully")
        return tokenizer
