from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def generate_alert():
    return {"message": "RAG alert endpoint"}
