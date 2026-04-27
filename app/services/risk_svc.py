"""Risk Analysis Service"""

from app.ml.knn.predictor import KNNRiskPredictor
from app.schemas.employee import RiskLabel
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskService:
    """Service for risk analysis operations"""
    
    def __init__(self, model_dir: Optional[str] = None):
        self.predictor = KNNRiskPredictor.get_instance(model_dir)
    
    def analyze_risk(
        self,
        user_id: str,
        features: Dict[str, float]
    ) -> dict:
        """
        Analyze risk level for employee
        
        Args:
            user_id: Employee UUID
            features: Dictionary containing:
                - kpi_completion_rate
                - checkin_delay_days
                - feedback_sentiment_score
                - objective_participation_ratio
        
        Returns:
            Dict with risk_label and risk_score
        """
        try:
            risk_label, risk_score = self.predictor.predict(features)
            
            return {
                "user_id": user_id,
                "risk_label": risk_label,
                "risk_score": float(risk_score),
                "features": features
            }
        except Exception as e:
            logger.error(f"Error analyzing risk for user {user_id}: {e}")
            raise
    
    def analyze_batch_risks(
        self,
        employees: list[dict]
    ) -> list[dict]:
        """
        Analyze risks for multiple employees
        
        Args:
            employees: List of dicts with user_id and features
        
        Returns:
            List of risk analysis results
        """
        results = []
        for employee in employees:
            user_id = employee.get('user_id')
            features = employee.get('features')
            
            if not user_id or not features:
                logger.warning(f"Skipping employee with missing user_id or features")
                continue
            
            result = self.analyze_risk(user_id, features)
            results.append(result)
        
        return results
    
    def get_risk_distribution(self, risk_analyses: list[dict]) -> dict:
        """
        Get distribution of risk levels
        
        Args:
            risk_analyses: List of risk analysis results
        
        Returns:
            Dict with counts for each risk level
        """
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for analysis in risk_analyses:
            label = analysis.get('risk_label')
            if label in distribution:
                distribution[label] += 1
        
        return distribution
