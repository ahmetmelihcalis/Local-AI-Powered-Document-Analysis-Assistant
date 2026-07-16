import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.database import DATABASE_PATH
from app.repositories.document_repository import get_chunks
from app.services.foundry_service import create_embeddings


DEFAULT_TOP_K = 4
MIN_SIMILARITY = 0.30
MAX_SCORE_DROP = 0.10
KEYWORD_WEIGHT = 0.12
BM25_K1 = 1.5
BM25_B = 0.75
HIGH_CONFIDENCE_SCORE = 0.50
MEDIUM_CONFIDENCE_SCORE = 0.35
MIN_RESULT_COUNT = 2
ADJACENT_CHUNK_BONUS = 0.10

KEYWORD_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
}


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


def _keyword_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in KEYWORD_STOP_WORDS
    ]


def _query_tokens(question: str) -> list[str]:
    return list(dict.fromkeys(_keyword_tokens(question)))


def is_list_question(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    list_terms = {"list", "what are", "which"}
    return any(term in normalized for term in list_terms) or (
        normalized.startswith("what ") and " are " in normalized
    )


def bm25_scores(question: str, documents: list[str]) -> np.ndarray:
    query_tokens = _query_tokens(question)
    if not query_tokens or not documents:
        return np.zeros(len(documents), dtype=np.float32)

    tokenized_documents = [_keyword_tokens(document) for document in documents]
    frequencies = [Counter(tokens) for tokens in tokenized_documents]
    document_lengths = np.asarray(
        [len(tokens) for tokens in tokenized_documents],
        dtype=np.float32,
    )
    average_length = float(document_lengths.mean()) or 1.0
    document_count = len(documents)
    scores = np.zeros(document_count, dtype=np.float32)

    for token in query_tokens:
        document_frequency = sum(token in frequency for frequency in frequencies)
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        for index, frequency in enumerate(frequencies):
            term_frequency = frequency.get(token, 0)
            if not term_frequency:
                continue

            length_normalization = 1 - BM25_B + BM25_B * (
                document_lengths[index] / average_length
            )
            scores[index] += inverse_document_frequency * (
                term_frequency * (BM25_K1 + 1)
                / (term_frequency + BM25_K1 * length_normalization)
            )

    return scores


def retrieve_relevant_chunks(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    max_score_drop: float = MAX_SCORE_DROP,
    database_path: Path = DATABASE_PATH,
    embedding_function: Callable[[list[str]], list[list[float]]] = create_embeddings,
) -> list[RetrievedChunk]:
    question = question.strip()

    if not question:
        raise ValueError("The question cannot be empty.")

    if top_k < 1:
        raise ValueError("Top-K must be at least 1.")

    if max_score_drop < 0:
        raise ValueError("Maximum score drop cannot be negative.")

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
    keyword_scores = bm25_scores(
        question,
        [
            "\n".join(part for part in (chunk["section"], chunk["content"]) if part)
            for chunk in chunks
        ],
    )
    maximum_keyword_score = float(keyword_scores.max())
    if maximum_keyword_score > 0:
        keyword_scores /= maximum_keyword_score

    base_ranking_scores = similarities + KEYWORD_WEIGHT * keyword_scores
    ranked_indices = np.argsort(base_ranking_scores)[::-1]
    best_score = float(base_ranking_scores[ranked_indices[0]])
    if best_score < min_similarity:
        return []

    accepted_score = max(min_similarity, best_score - max_score_drop)
    list_question = is_list_question(question)
    if list_question:
        result_limit = top_k
    elif best_score >= HIGH_CONFIDENCE_SCORE:
        result_limit = min(top_k, 2)
    elif best_score >= MEDIUM_CONFIDENCE_SCORE:
        result_limit = min(top_k, 3)
    else:
        result_limit = top_k

    ranking_scores = base_ranking_scores.copy()
    top_index = int(ranked_indices[0])
    top_chunk = chunks[top_index]
    related_indices: set[int] = set()
    if top_chunk["section"]:
        for index, chunk in enumerate(chunks):
            if (
                index == top_index
                or chunk["document_id"] != top_chunk["document_id"]
                or chunk["section"] != top_chunk["section"]
            ):
                continue

            distance = abs(chunk["chunk_index"] - top_chunk["chunk_index"])
            if distance == 1 or list_question:
                ranking_scores[index] += ADJACENT_CHUNK_BONUS / distance
                related_indices.add(index)

        remaining_indices = [
            int(index) for index in ranked_indices if int(index) != top_index
        ]
        remaining_indices.sort(
            key=lambda index: float(ranking_scores[index]),
            reverse=True,
        )
        if list_question:
            section_indices = sorted(
                related_indices,
                key=lambda index: abs(
                    chunks[index]["chunk_index"] - top_chunk["chunk_index"]
                ),
            )
            remaining_indices = [
                index for index in remaining_indices if index not in related_indices
            ]
            ranked_indices = np.asarray(
                [top_index, *section_indices, *remaining_indices],
                dtype=int,
            )
        else:
            ranked_indices = np.asarray([top_index, *remaining_indices], dtype=int)

    results: list[RetrievedChunk] = []
    for index in ranked_indices:
        score = float(similarities[index])
        if (
            float(base_ranking_scores[index]) < accepted_score
            and len(results) >= min(MIN_RESULT_COUNT, result_limit)
            and not (list_question and int(index) in related_indices)
        ):
            break

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

        if len(results) == result_limit:
            break

    return results
