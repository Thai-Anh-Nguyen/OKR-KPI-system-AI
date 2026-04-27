import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from app.rag.retriever import retrieve_hr_context

logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=2)


def _run_rag_generation(user_id: str, risk_score: float) -> dict:
    """
    Synchronous RAG pipeline:
    1. Retrieve relevant HR policies from pgvector.
    2. Send context + employee metrics to an LLM (Claude/OpenAI).
    3. Return a structured alert JSON.
    """
    # Step 1: Retrieve context from vector DB
    hr_context = retrieve_hr_context(user_id, risk_score)

    # Step 2: Build the LLM prompt and call the model
    # TODO: Integrate with Claude/OpenAI API
    # For now, return a structured stub based on the retrieved context
    alert = {
        "alert_title": f"Risk Alert for Employee {user_id[:8]}",
        "alert_message": (
            f"Employee has a risk score of {risk_score:.2f}. "
            f"Context retrieved: {hr_context.get('summary', 'N/A')}. "
            "Recommend manager review."
        ),
        "recommended_action": "Schedule a 1-on-1 meeting to discuss workload and engagement.",
    }
    return alert


async def generate_alert_async(user_id: str, risk_score: float) -> dict:
    """
    Generates an AI alert using RAG if risk_score >= 0.5.
    """
    if risk_score < 0.5:
        logger.info(f"Risk score {risk_score} below threshold for user {user_id}. No alert generated.")
        return {
            "alert_title": "No Alert",
            "alert_message": "Risk score is below the threshold.",
            "recommended_action": "No action needed.",
        }

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _run_rag_generation, user_id, risk_score)

    logger.info(f"RAG alert generated for user {user_id}")
    return result
