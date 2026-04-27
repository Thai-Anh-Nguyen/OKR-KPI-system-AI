import logging
from typing import List

logger = logging.getLogger(__name__)

# Global variable to hold the trained KNN model
_knn_model = None


def load_knn_model():
    """
    Loads or initializes the KNN model.
    Called during FastAPI startup if the model file exists,
    or trained on-the-fly from the employee_features table.
    """
    global _knn_model
    # TODO: Load a persisted sklearn model from disk, or train from DB data
    # from joblib import load
    # _knn_model = load("path/to/knn_model.joblib")
    logger.info("KNN model loader called (not yet implemented).")


def predict_risk_label(features: List[float]) -> dict:
    """
    Predicts a risk label using KNN clustering.

    Args:
        features: A list of numerical features from the ETL pipeline.

    Returns:
        dict with 'label' (LOW|MEDIUM|HIGH) and 'score' (0.0-1.0).
    """
    # TODO: Replace with actual sklearn prediction
    # if _knn_model is None:
    #     raise RuntimeError("KNN model is not loaded.")
    # prediction = _knn_model.predict([features])

    # Stub: simple threshold-based logic until the real model is trained
    avg_feature = sum(features) / len(features) if features else 0.0

    if avg_feature >= 0.7:
        return {"label": "HIGH", "score": round(avg_feature, 4)}
    elif avg_feature >= 0.4:
        return {"label": "MEDIUM", "score": round(avg_feature, 4)}
    else:
        return {"label": "LOW", "score": round(avg_feature, 4)}
