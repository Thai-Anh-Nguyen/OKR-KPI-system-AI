from fastapi import APIRouter, HTTPException
from app.models.schemas import RiskRequest, RiskResponse
import app.services.risk_svc as risk_service

router = APIRouter()

@router.post("/risk", response_model=RiskResponse)
async def analyze_risk(request: RiskRequest):
    """
    Evaluates employee risk using KNN clustering based on ETL features.
    Called by the Node.js ETL job (behaviorAnalysis.job.js).
    """
    try:
        result = await risk_service.analyze_risk_async(str(request.user_id), request.features)
        return RiskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
