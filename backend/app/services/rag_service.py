import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.database import DATABASE_PATH
from app.services.foundry_service import create_embeddings, generate_chat
from app.services.retrieval import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY,
    RetrievedChunk,
    retrieve_relevant_chunks,
)


INSUFFICIENT_ANSWER = (
    "The uploaded documents do not contain enough information to answer this question."
)

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Use only the provided context. Do not add "
    "or infer anything that is not explicitly stated in the context. Support every "
    "claim with source numbers such as [1] and [2]. Do not repeat information. Write "
    "only source numbers that appear in the context. Write at most three short "
    "sentences. If the information is insufficient, only say so. Answer in English."
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]
    retrieval_count: int


def _format_context(sources: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []

    for number, source in enumerate(sources, start=1):
        metadata = [f"file={source.file_name}"]
        if source.page_number is not None:
            metadata.append(f"page={source.page_number}")
        if source.section:
            metadata.append(f"section={source.section}")

        context_blocks.append(
            f"[SOURCE {number} | {', '.join(metadata)}]\n{source.content}"
        )

    return "\n\n".join(context_blocks)


def _remove_invalid_citations(answer: str, source_count: int) -> str:
    def replace_citation(match: re.Match[str]) -> str:
        source_number = int(match.group(1))
        return match.group(0) if 1 <= source_number <= source_count else ""

    cleaned = re.sub(r"\[(\d+)\]", replace_citation, answer)
    return " ".join(cleaned.split())


def answer_question(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    database_path: Path = DATABASE_PATH,
    embedding_function: Callable[[list[str]], list[list[float]]] = create_embeddings,
    chat_function: Callable[[list[dict[str, str]]], str] = generate_chat,
) -> RagAnswer:
    sources = retrieve_relevant_chunks(
        question,
        top_k=top_k,
        min_similarity=min_similarity,
        database_path=database_path,
        embedding_function=embedding_function,
    )

    if not sources:
        return RagAnswer(
            answer=INSUFFICIENT_ANSWER,
            sources=[],
            retrieval_count=0,
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{_format_context(sources)}\n\n"
                f"QUESTION:\n{question}\n\n"
                "Answer the question directly and obey every system instruction."
            ),
        },
    ]
    answer = _remove_invalid_citations(chat_function(messages), len(sources))
    return RagAnswer(
        answer=answer,
        sources=sources,
        retrieval_count=len(sources),
    )
