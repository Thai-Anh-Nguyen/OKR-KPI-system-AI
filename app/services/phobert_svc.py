from app.ml.phobert.loader import get_sentiment_pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Use a thread pool to avoid blocking the FastAPI event loop during heavy CPU inference
executor = ThreadPoolExecutor(max_workers=4)

def _run_inference(text: str) -> dict:
    pipeline = get_sentiment_pipeline()
    
    # The pipeline handles tokenization. We truncate arbitrarily long text to avoid crashes
    # PhoBERT handles sequences up to 256 tokens well.
    truncated_text = text[:800] 
    
    result = pipeline(truncated_text)[0]
    return result

async def analyze_sentiment_async(text: str) -> dict:
    # Run the blocking inference in a separate thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _run_inference, text)
    
    original_label = result['label'] # Usually 'POS', 'NEG', 'NEU'
    score = result['score']
    
    # Map to your Database Schema Enum
    mapping = {
        'POS': 'POSITIVE',
        'NEG': 'NEGATIVE',
        'NEU': 'NEUTRAL'
    }
    
    sentiment = mapping.get(original_label, 'UNKNOWN')
    
    return {
        "sentiment": sentiment,
        "score": score,
        "original_label": original_label
    }
