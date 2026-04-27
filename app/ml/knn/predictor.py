"""KNN Risk Model Predictor"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Literal, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

RiskLabel = Literal["low", "medium", "high"]


class KNNRiskPredictor:
    """Wrapper for KNN risk classification model"""
    
    _instance = None
    _model = None
    _feature_cols = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize KNN predictor
        
        Args:
            model_dir: Directory containing knn_risk_model.pkl and feature_columns.pkl
        """
        if self._model is None:
            if model_dir is None:
                model_dir = os.path.join(os.path.dirname(__file__), '../../models')
            self._load_model(model_dir)
    
    def _load_model(self, model_dir: str):
        """Load model and feature columns from disk"""
        model_path = os.path.join(model_dir, 'knn_risk_model.pkl')
        cols_path = os.path.join(model_dir, 'feature_columns.pkl')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Using demo mode.")
            self._model = None
            self._feature_cols = [
                'kpi_completion_rate',
                'checkin_delay_days',
                'feedback_sentiment_score',
                'objective_participation_ratio',
            ]
            return
        
        try:
            self._model = joblib.load(model_path)
            self._feature_cols = joblib.load(cols_path)
            logger.info(f"KNN model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load KNN model: {e}")
            self._model = None
            self._feature_cols = [
                'kpi_completion_rate',
                'checkin_delay_days',
                'feedback_sentiment_score',
                'objective_participation_ratio',
            ]
    
    def predict(
        self,
        features: Dict[str, float]
    ) -> tuple[RiskLabel, float]:
        """
        Predict risk label and score for employee
        
        Args:
            features: Dictionary with keys:
                - kpi_completion_rate (0.0-1.0)
                - checkin_delay_days (float)
                - feedback_sentiment_score (-1.0 to 1.0)
                - objective_participation_ratio (float)
        
        Returns:
            Tuple of (risk_label, risk_score)
        """
        # Validate input
        required_keys = set(self._feature_cols)
        provided_keys = set(features.keys())
        
        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(f"Missing features: {missing}")
        
        # Prepare feature vector
        X = np.array([[features[col] for col in self._feature_cols]])
        
        if self._model is None:
            # Rule-based fallback
            return self._rule_based_predict(features)
        
        try:
            # Use trained model
            label = self._model.predict(X)[0]
            
            # Convert label to numeric score
            label_to_score = {'low': 0.1, 'medium': 0.5, 'high': 1.0}
            risk_score = label_to_score.get(label, 0.5)
            
            return label, risk_score
            
        except Exception as e:
            logger.error(f"Error during risk prediction: {e}")
            # Fall back to rule-based
            return self._rule_based_predict(features)
    
    def predict_batch(
        self,
        features_list: List[Dict[str, float]]
    ) -> List[tuple[str, float]]:
        """
        Predict risk for multiple employees
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of (risk_label, risk_score) tuples
        """
        results = []
        for features in features_list:
            label, score = self.predict(features)
            results.append((label, score))
        return results
    
    def _rule_based_predict(self, features: Dict[str, float]) -> tuple[RiskLabel, float]:
        """
        Rule-based risk prediction for demo/fallback
        """
        kpi = features.get('kpi_completion_rate', 0.5)
        delay = features.get('checkin_delay_days', 5.0)
        sentiment = features.get('feedback_sentiment_score', 0.0)
        obj_ratio = features.get('objective_participation_ratio', 1.0)
        
        risk_score = 0.0
        
        # KPI below 0.6 is risky
        if kpi < 0.5:
            risk_score += 0.30
        elif kpi < 0.7:
            risk_score += 0.15
        
        # High delay is risky
        if delay > 10:
            risk_score += 0.25
        elif delay > 7:
            risk_score += 0.12
        
        # Negative sentiment is risky
        if sentiment < -0.5:
            risk_score += 0.25
        elif sentiment < 0:
            risk_score += 0.12
        
        # Low objective participation is risky
        if obj_ratio < 0.5:
            risk_score += 0.20
        elif obj_ratio < 1.0:
            risk_score += 0.10
        
        # Classify based on score
        if risk_score >= 0.40:
            label = 'high'
        elif risk_score >= 0.18:
            label = 'medium'
        else:
            label = 'low'
        
        return label, min(risk_score, 1.0)
    
    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance
