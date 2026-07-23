import logging
from time import perf_counter
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.services.rag_service import answer_question


router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        return question


class SourceResponse(BaseModel):
    documentId: int
    fileName: str
    page: int | None
    section: str | None
    article: str | None
    paragraph: str | None
    point: str | None
    subpoint: str | None
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    retrievalCount: int
    durationMs: int
    timings: dict[str, int]


def _create_excerpt(content: str, max_length: int = 320) -> str:
    excerpt = " ".join(content.split())
    if len(excerpt) <= max_length:
        return excerpt
    return f"{excerpt[: max_length - 1].rstrip()}…"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    started_at = perf_counter()

    try:
        result = await run_in_threadpool(
            answer_question,
            request.question,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Local RAG answer generation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local AI service is currently unavailable.",
        ) from error

    duration_ms = round((perf_counter() - started_at) * 1000)
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                documentId=source.document_id,
                fileName=source.file_name,
                page=source.page_number,
                section=source.section,
                article=source.article,
                paragraph=source.paragraph,
                point=source.point,
                subpoint=source.subpoint,
                excerpt=_create_excerpt(source.content),
                score=round(source.score, 4),
            )
            for source in result.sources
        ],
        retrievalCount=result.retrieval_count,
        durationMs=duration_ms,
        timings=result.timings,
    )
