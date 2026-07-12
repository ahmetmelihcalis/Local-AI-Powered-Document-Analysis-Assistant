from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.database import initialize_database
from app.services.foundry_service import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    foundry_status,
    test_chat,
)


class TestChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    initialize_database()
    yield


app = FastAPI(title="Local RAG Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "foundryLocal": foundry_status(),
        "chatModel": CHAT_MODEL,
        "embeddingModel": EMBEDDING_MODEL,
    }


@app.post("/api/chat/test")
def chat_test(request: TestChatRequest) -> dict[str, str]:
    try:
        answer = test_chat(request.message)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"answer": answer, "model": CHAT_MODEL}
