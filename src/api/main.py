from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.answer_generator import AnswerGenerator


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="ZX Bank Enterprise Knowledge Assistant API",
    description="API for the ZX Bank RAG knowledge assistant",
    version="1.0.0",
)


# --------------------------------------------------
# LOAD RAG SYSTEM
# --------------------------------------------------

rag = AnswerGenerator(top_k=5)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    conversation_history: list[dict] = []


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "ZX Bank Enterprise Knowledge Assistant",
    }


# --------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    result = rag.generate_answer(
        question=request.question,
        conversation_history=request.conversation_history,
    )

    return {
        "question": result["question"],
        "search_question": result["search_question"],
        "answer": result["answer"],
        "sources": result["sources"],
    }