import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.generation.answer_generator import AnswerGenerator
from src.logging_config import setup_logging


# ============================================================
# LOGGING
# ============================================================

setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ZX Bank Enterprise Knowledge Assistant API",
    description="API for the ZX Bank RAG knowledge assistant",
    version="1.0.0",
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# RAG ENGINE
# ============================================================

rag = AnswerGenerator(top_k=5)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question for the knowledge assistant.",
    )

    conversation_history: list[dict] = Field(
        default_factory=list,
        max_length=20,
        description="Previous conversation messages.",
    )

    @field_validator("question", mode="before")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    logger.info("Health check requested")

    return {
        "status": "healthy",
        "service": "ZX Bank Enterprise Knowledge Assistant",
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

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

    except Exception as exc:

        logger.exception(
            "Error while processing question: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        )