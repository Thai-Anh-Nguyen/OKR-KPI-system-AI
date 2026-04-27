from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def analyze_sentiment():
    return {"message": "Sentiment analysis endpoint"}
