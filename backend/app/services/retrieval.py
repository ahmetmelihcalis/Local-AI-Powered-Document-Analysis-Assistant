from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.database import DATABASE_PATH
from app.repositories.document_repository import get_chunks
from app.services.foundry_service import create_embeddings


DEFAULT_TOP_K = 4
MIN_SIMILARITY = 0.30


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    file_name: str
    content: str
    page_number: int | None
    section: str | None
    article: str | None
    paragraph: str | None
    point: str | None
    subpoint: str | None
    score: float


def cosine_similarities(
    query_embedding: list[float] | np.ndarray,
    chunk_embeddings: list[np.ndarray],
) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)

    if query.ndim != 1 or query.size == 0:
        raise ValueError("The query embedding must be a non-empty vector.")

    if not chunk_embeddings:
        return np.array([], dtype=np.float32)

    matrix = np.vstack(chunk_embeddings).astype(np.float32, copy=False)
    if matrix.shape[1] != query.size:
        raise ValueError("Query and chunk embeddings must have the same dimensions.")

    query_norm = np.linalg.norm(query)
    chunk_norms = np.linalg.norm(matrix, axis=1)
    denominators = chunk_norms * query_norm
    similarities = np.full(matrix.shape[0], -1.0, dtype=np.float32)

    np.divide(
        matrix @ query,
        denominators,
        out=similarities,
        where=denominators > 0,
    )
    return similarities


def retrieve_relevant_chunks(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    database_path: Path = DATABASE_PATH,
    embedding_function: Callable[[list[str]], list[list[float]]] = create_embeddings,
) -> list[RetrievedChunk]:
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    if top_k < 1:
        raise ValueError("Top-K must be at least 1.")

    chunks = get_chunks(database_path=database_path)
    if not chunks:
        return []

    response = embedding_function([question])
    if len(response) != 1:
        raise RuntimeError("The embedding model returned an unexpected result count.")

    similarities = cosine_similarities(
        response[0],
        [chunk["embedding"] for chunk in chunks],
    )
    ranked_indices = np.argsort(similarities)[::-1]

    results: list[RetrievedChunk] = []
    for index in ranked_indices:
        score = float(similarities[index])
        if score < min_similarity:
            continue

        chunk = chunks[int(index)]
        results.append(
            RetrievedChunk(
                chunk_id=chunk["id"],
                document_id=chunk["document_id"],
                file_name=chunk["original_name"],
                content=chunk["content"],
                page_number=chunk["page_number"],
                section=chunk["section"],
                article=chunk["article"],
                paragraph=chunk["paragraph"],
                point=chunk["point"],
                subpoint=chunk["subpoint"],
                score=score,
            )
        )

        if len(results) == top_k:
            break

    return results
