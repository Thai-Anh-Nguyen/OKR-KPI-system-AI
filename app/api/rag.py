from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGRequest, RAGResponse
import app.services.rag_svc as rag_service

router = APIRouter()

@router.post("/rag", response_model=RAGResponse)
async def generate_alert(request: RAGRequest):
    """
    Generates an AI Alert using Retrieval-Augmented Generation (RAG).
    Auto-triggered when risk_score >= 0.5. Called by the Node.js ETL job.
    """
    try:
        result = await rag_service.generate_alert_async(str(request.user_id), request.risk_score)
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
