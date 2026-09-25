
"""
Zepto Support Assistant FastAPI API
====================================

Run locally with:

    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoint:

    POST /ask

Request:
    {
        "query": "How long does delivery take?"
    }

Response:
    {
        "answer": "...",
        "sources": ["doc_01", "doc_02", "doc_04"],
        "confidence": 0.4154
    }
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_assistant import ask


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Local-first Zepto policy assistant using "
        "LangGraph, ChromaDB and SentenceTransformers."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class AskRequest(BaseModel):
    """
    Request body for POST /ask.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="User's Zepto policy question.",
    )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class AskResponse(BaseModel):
    """
    Structured response returned by POST /ask.
    """

    answer: str

    sources: list[str]

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    """
    Basic API health endpoint.
    """

    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "endpoint": "/ask",
        "mock_llm": True,
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_endpoint(request: AskRequest):
    """
    Answer a Zepto policy question.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        result = ask(query)

        return AskResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Support assistant error: {exc}",
        ) from exc


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
