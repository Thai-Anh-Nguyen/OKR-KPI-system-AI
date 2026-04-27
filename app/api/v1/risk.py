"""Risk Analysis API Endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List
from app.schemas.employee import RiskAnalysisRequest, RiskAnalysisResponse
from app.services.risk_svc import RiskService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_risk_service():
    """Dependency injection for risk service"""
    return RiskService()


@router.post("/", response_model=RiskAnalysisResponse)
async def analyze_risk(
    request: RiskAnalysisRequest,
    service: RiskService = Depends(get_risk_service)
):
    """
    Analyze risk level for an employee
    
    - **user_id**: Employee UUID
    - **features**: Employee feature vector containing:
        - kpi_completion_rate (0.0-1.0)
        - checkin_delay_days (float)
        - feedback_sentiment_score (-1.0 to 1.0)
        - objective_participation_ratio (float)
    
    Returns risk classification (low/medium/high) and numerical risk score
    """
    try:
        result = service.analyze_risk(
            request.user_id,
            request.features.dict()
        )
        return RiskAnalysisResponse(**result)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing risk: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch")
async def analyze_risks_batch(
    requests: List[RiskAnalysisRequest],
    service: RiskService = Depends(get_risk_service)
):
    """
    Analyze risks for multiple employees in batch
    
    Returns list of risk analysis results
    """
    try:
        employees = [
            {
                'user_id': req.user_id,
                'features': req.features.dict()
            }
            for req in requests
        ]
        results = service.analyze_batch_risks(employees)
        return {
            "total": len(results),
            "analyses": results,
            "distribution": service.get_risk_distribution(results)
        }
    except Exception as e:
        logger.error(f"Error in batch risk analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def risk_health():
    """Health check for risk analysis service"""
    return {"status": "ok", "service": "risk-analysis"}
