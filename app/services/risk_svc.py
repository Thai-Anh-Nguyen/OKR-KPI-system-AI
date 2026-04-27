import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List
from app.ml.knn.predictor import predict_risk_label

logger = logging.getLogger(__name__)

# Use a thread pool to avoid blocking the FastAPI event loop during KNN inference
executor = ThreadPoolExecutor(max_workers=4)


def _run_knn_inference(features: List[float]) -> dict:
    """
    Runs KNN prediction synchronously.
    This is offloaded to a thread pool from the async handler.
    """
    knn_risk_label = predict_risk_label(features)
    return knn_risk_label


async def analyze_risk_async(user_id: str, features: List[float]) -> dict:
    """
    Analyzes risk for a user based on their ETL features using KNN clustering.
    Returns a risk label and a normalized risk score.
    """
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _run_knn_inference, features)

    knn_risk_label = result.get("label", "LOW")
    risk_score = result.get("score", 0.0)

    logger.info(f"Risk analysis for user {user_id}: label={knn_risk_label}, score={risk_score}")

    return {
        "knn_risk_label": knn_risk_label,
        "risk_score": risk_score,
    }
