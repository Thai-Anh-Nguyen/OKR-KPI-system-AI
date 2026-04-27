"""RAG Alert Generation API Endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.schemas.rag import AlertRequest, AlertResponse
from app.services.rag_svc import RAGAlertService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service():
    """Dependency injection for RAG service"""
    return RAGAlertService()


@router.post("/", response_model=AlertResponse)
async def generate_alert(
    request: AlertRequest,
    service: RAGAlertService = Depends(get_rag_service)
):
    """
    Generate HR alert using RAG pipeline
    
    - **user_id**: Employee UUID
    - **risk_score**: Risk score (0.0-1.0)
    - **risk_label**: Risk classification (low/medium/high)
    - **context**: Optional additional context for LLM
    
    Returns structured alert with recommendations
    """
    try:
        result = service.generate_alert(
            user_id=request.user_id,
            risk_label=request.risk_label,
            risk_score=request.risk_score,
            context=request.context
        )
        return AlertResponse(**result)
    except Exception as e:
        logger.error(f"Error generating alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch")
async def generate_alerts_batch(
    risk_analyses: List[dict],
    service: RAGAlertService = Depends(get_rag_service)
):
    """
    Generate alerts for multiple risk analyses
    
    Only generates alerts for medium/high risk (score >= 0.18)
    
    Returns list of generated alerts
    """
    try:
        alerts = await service.generate_alerts_batch(risk_analyses)
        return {
            "total": len(alerts),
            "alerts": alerts
        }
    except Exception as e:
        logger.error(f"Error in batch alert generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def rag_health():
    """Health check for RAG alert service"""
    return {"status": "ok", "service": "rag-alert-generation"}
