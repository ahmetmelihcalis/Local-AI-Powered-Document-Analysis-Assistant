import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from app.database import DATABASE_PATH
from app.repositories.document_repository import get_chunks
from app.services.foundry_service import create_embeddings


DEFAULT_TOP_K = 4
MIN_SIMILARITY = 0.30
MAX_SCORE_DROP = 0.10
KEYWORD_WEIGHT = 0.12
DEFINITION_MATCH_BONUS = 0.15
MIN_RELEVANCE_COSINE = 0.55
HIGH_RELEVANCE_COSINE = 0.72
MIN_QUERY_COVERAGE = 0.30
BM25_K1 = 1.5
BM25_B = 0.75
HIGH_CONFIDENCE_SCORE = 0.50
MEDIUM_CONFIDENCE_SCORE = 0.35
MIN_RESULT_COUNT = 2
ADJACENT_CHUNK_BONUS = 0.10
MAX_COMPLETE_SCOPE_CHUNKS = 20

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


class QuestionScope(StrEnum):
    DEFINITION = "definition"
    FOCUSED = "focused"
    COMPLETE_LIST = "complete_list"

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


def classify_question_scope(
    question: str,
    top_point: str | None,
) -> QuestionScope:
    if is_definition_question(question):
        return QuestionScope.DEFINITION

    normalized = " ".join(question.casefold().split())
    complete_scope_terms = (
        "what obligations",
        "which obligations",
        "what requirements",
        "which requirements",
        "what duties",
        "which duties",
    )
    if normalized.startswith(complete_scope_terms):
        return QuestionScope.COMPLETE_LIST
    if top_point is not None:
        return QuestionScope.FOCUSED
    if is_list_question(question):
        return QuestionScope.COMPLETE_LIST
    return QuestionScope.FOCUSED


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


def _passes_relevance_gate(
    question: str,
    document: str,
    cosine_score: float,
    definition_score: float,
) -> bool:
    if definition_score > 0 or cosine_score >= HIGH_RELEVANCE_COSINE:
        return True
    if cosine_score < MIN_RELEVANCE_COSINE:
        return False

    query_tokens = set(_query_tokens(question))
    if not query_tokens:
        return False
    document_tokens = set(_keyword_tokens(document))
    coverage = len(query_tokens & document_tokens) / len(query_tokens)
    return coverage >= MIN_QUERY_COVERAGE


def _definition_subject(question: str) -> str | None:
    normalized = " ".join(question.casefold().strip(" ?.!\t\n").split())
    if normalized.startswith("what is "):
        subject = normalized.removeprefix("what is ")
    else:
        return None

    for article in ("a ", "an ", "the "):
        if subject.startswith(article):
            subject = subject.removeprefix(article)
            break
    return subject


def is_definition_question(question: str) -> bool:
    return _definition_subject(question) is not None


def _definition_scores(question: str, documents: list[str]) -> np.ndarray:
    subject = _definition_subject(question)
    scores = np.zeros(len(documents), dtype=np.float32)
    if not subject:
        return scores

    pattern = re.compile(
        rf"[‘’'\"]?{re.escape(subject)}[‘’'\"]?\s+means\b",
        re.IGNORECASE,
    )
    for index, document in enumerate(documents):
        if pattern.search(document):
            scores[index] = 1.0
    return scores


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


def _article_heading_index(
    question: str,
    chunks: list[dict],
    ranking_scores: np.ndarray,
) -> int | None:
    query_tokens = set(_query_tokens(question))
    candidates: list[tuple[float, float, int]] = []

    for index, chunk in enumerate(chunks):
        if (
            chunk["article"] is None
            or chunk["paragraph"] is not None
            or chunk["point"] is not None
            or chunk["subpoint"] is not None
        ):
            continue

        heading_tokens = set(
            _keyword_tokens(
                "\n".join(
                    part for part in (chunk["section"], chunk["content"]) if part
                )
            )
        )
        coverage = (
            len(query_tokens & heading_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        candidates.append((coverage, float(ranking_scores[index]), index))

    if not candidates:
        return None

    coverage, _, index = max(candidates)
    return index if coverage >= 0.5 else None


def _definition_indices(
    definition_scores: np.ndarray,
    ranking_scores: np.ndarray,
) -> list[int]:
    matching_indices = np.flatnonzero(definition_scores > 0)
    if matching_indices.size == 0:
        return [int(np.argmax(ranking_scores))]
    return [max(matching_indices, key=lambda index: float(ranking_scores[index])).item()]


def _focused_clause_indices(
    chunks: list[dict],
    anchor_index: int,
) -> list[int]:
    anchor = chunks[anchor_index]
    hierarchy = (
        anchor["document_id"],
        anchor["article"],
        anchor["paragraph"],
        anchor["point"],
        anchor["subpoint"],
    )
    if anchor["article"] is None:
        return [anchor_index]

    return [
        index
        for index, chunk in enumerate(chunks)
        if (
            chunk["document_id"],
            chunk["article"],
            chunk["paragraph"],
            chunk["point"],
            chunk["subpoint"],
        )
        == hierarchy
    ]


def _complete_scope_indices(
    chunks: list[dict],
    anchor_index: int,
    original_top_index: int,
) -> list[int]:
    anchor = chunks[anchor_index]
    document_id = anchor["document_id"]
    article = anchor["article"]
    paragraph = None

    original_top = chunks[original_top_index]
    if (
        original_top["document_id"] == document_id
        and original_top["article"] == article
        and original_top["paragraph"] is not None
        and original_top["point"] is None
    ):
        paragraph = original_top["paragraph"]

    indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["document_id"] == document_id
        and chunk["article"] == article
        and (paragraph is None or chunk["paragraph"] == paragraph)
    ]
    substantive_indices = [
        index
        for index in indices
        if chunks[index]["paragraph"] is not None or chunks[index]["point"] is not None
    ]
    return (substantive_indices or indices)[:MAX_COMPLETE_SCOPE_CHUNKS]


def _retrieved_chunk(
    chunk: dict,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
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

    definition_scores = _definition_scores(
        question,
        [chunk["content"] for chunk in chunks],
    )
    base_ranking_scores = (
        similarities
        + KEYWORD_WEIGHT * keyword_scores
        + DEFINITION_MATCH_BONUS * definition_scores
    )
    ranked_indices = np.argsort(base_ranking_scores)[::-1]
    best_score = float(base_ranking_scores[ranked_indices[0]])
    if best_score < min_similarity:
        return []

    original_top_index = int(ranked_indices[0])
    top_document = "\n".join(
        part
        for part in (
            chunks[original_top_index]["section"],
            chunks[original_top_index]["content"],
        )
        if part
    )
    if not _passes_relevance_gate(
        question,
        top_document,
        float(similarities[original_top_index]),
        float(definition_scores[original_top_index]),
    ):
        return []

    scope = classify_question_scope(question, chunks[original_top_index]["point"])
    if scope == QuestionScope.DEFINITION:
        selected_indices = _definition_indices(definition_scores, base_ranking_scores)
        return [
            _retrieved_chunk(chunks[index], float(similarities[index]))
            for index in selected_indices
        ]

    if scope == QuestionScope.COMPLETE_LIST:
        heading_index = _article_heading_index(
            question,
            chunks,
            base_ranking_scores,
        )
        anchor_index = (
            heading_index if heading_index is not None else original_top_index
        )
        if chunks[anchor_index]["article"] is not None:
            selected_indices = _complete_scope_indices(
                chunks,
                anchor_index,
                original_top_index,
            )
            return [
                _retrieved_chunk(chunks[index], float(similarities[index]))
                for index in selected_indices
            ]

    if chunks[original_top_index]["article"] is not None:
        if chunks[original_top_index]["paragraph"] is None:
            selected_indices = _complete_scope_indices(
                chunks,
                original_top_index,
                original_top_index,
            )
        else:
            selected_indices = _focused_clause_indices(chunks, original_top_index)
        return [
            _retrieved_chunk(chunks[index], float(similarities[index]))
            for index in selected_indices
        ]

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
