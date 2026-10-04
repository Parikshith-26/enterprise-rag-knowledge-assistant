import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generation.answer_generator import AnswerGenerator
from src.logging_config import setup_logging


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="ZX Bank Enterprise Knowledge Assistant API",
    description="API for the ZX Bank RAG knowledge assistant",
    version="1.0.0",
)


# ---------------------------------------------------------
# RAG engine
# ---------------------------------------------------------

rag = AnswerGenerator(top_k=5)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    conversation_history: list[dict] = []


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    logger.info("Health check requested")

    return {
        "status": "healthy",
        "service": "ZX Bank Enterprise Knowledge Assistant",
    }


# ---------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    logger.info(
        "Question received | length=%d | history_items=%d",
        len(request.question),
        len(request.conversation_history),
    )

    try:

        result = rag.generate_answer(
            question=request.question,
            conversation_history=request.conversation_history,
        )

        logger.info(
            "Question processed successfully | sources=%d",
            len(result.get("sources", [])),
        )

        return {
            "question": result["question"],
            "search_question": result["search_question"],
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception:
        logger.exception("Error while processing question")

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        )