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
from app.services.question_types import (
    QuestionType,
    build_retrieval_plan,
    classify_question_type,
)


DEFAULT_TOP_K = 4
MIN_SIMILARITY = 0.30
MAX_SCORE_DROP = 0.10
KEYWORD_WEIGHT = 0.12
QUERY_EXPANSION_WEIGHT = 0.08
COMPARISON_KEYWORD_WEIGHT = 0.25
PHRASE_MATCH_WEIGHT = 0.20
DEFINITION_MATCH_BONUS = 0.15
MIN_RELEVANCE_COSINE = 0.55
HIGH_RELEVANCE_COSINE = 0.68
MIN_QUERY_COVERAGE = 0.30
BM25_K1 = 1.5
BM25_B = 0.75
HIGH_CONFIDENCE_SCORE = 0.50
MEDIUM_CONFIDENCE_SCORE = 0.35
MIN_RESULT_COUNT = 2
ADJACENT_CHUNK_BONUS = 0.10
UNSECTIONED_CONTEXT_WINDOW = 2
LEGAL_HIERARCHY_CONTEXT_WEIGHT = 0.20
HIERARCHY_CONTEXT_ANCHOR_MIN_COVERAGE = 0.60
MAX_COMPLETE_SCOPE_CHUNKS = 20
PARAGRAPH_QUERY_COVERAGE_WEIGHT = 0.12
PARAGRAPH_CONTRAST_BONUS = 0.12
PARAGRAPH_TIMING_BONUS = 0.16
PARAGRAPH_CONTENT_BONUS = 0.18
PARAGRAPH_CONDITION_BONUS = 0.16
PSEUDO_RELEVANCE_CHUNKS = 4
MAX_EXPANSION_TERMS = 8
MIN_EXPANSION_DOCUMENTS = 2
SEMANTIC_OVERRIDE_MARGIN = 0.06
LEXICAL_ANCHOR_MIN_SCORE = 0.75
LEXICAL_ANCHOR_MIN_COVERAGE = 0.60
MAX_COMPARISON_DOCUMENTS = 3
MAX_COMPARISON_CHUNKS_PER_DOCUMENT = 2
DOCUMENT_NAME_MATCH_BONUS = 0.18
NORMATIVE_CLAUSE_BONUS = 0.05
OPERATIVE_CLAUSE_BONUS = 0.15
PREAMBLE_PENALTY = 0.12
MIN_SUPPLEMENTAL_QUERY_COVERAGE = 0.60
QUESTION_TYPE_MATCH_BONUS = 0.10
REFERENCE_CLAUSE_PENALTY = 0.12
DEFINITION_CLAUSE_PENALTY = 0.10

COMPARISON_PATTERNS = (
    re.compile(r"\bcompar(?:e|ed|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bdiffer(?:s|ed|ent|ently|ence|ences)?\b", re.IGNORECASE),
    re.compile(r"\b(?:versus|vs\.?)\b", re.IGNORECASE),
)

ARTICLE_CITATION = re.compile(
    r"\bArticle\s+(\d+[a-z]?)(?:\s*\(([0-9]+)\))?",
    re.IGNORECASE,
)

EXTERNAL_REGULATION_CITATION = re.compile(
    r"\bArticle\s+(\d+[a-z]?)\s+of\s+Regulation\s+"
    r"\(EU\)\s+(\d{4})\s*/\s*0*(\d+)",
    re.IGNORECASE,
)

CONTRAST_TERMS = {
    "except",
    "exception",
    "exempt",
    "not",
    "unless",
    "without",
}
TIMING_TEXT_TERMS = {"when", "within", "delay", "hours", "days", "before", "after"}
CONTENT_TEXT_TERMS = {"contain", "contains", "include", "includes", "information", "details"}
CONDITION_TEXT_TERMS = {"condition", "conditions", "where", "unless", "if"}
REFERENCE_CLAUSE = re.compile(
    r"^\s*(?:\(?\d+[.)]\s*)?(?:paragraphs?|articles?|sections?)\s+\d+"
    r".*\b(?:refer(?:red)? to|set out in|shall not affect|without prejudice)\b",
    re.IGNORECASE | re.DOTALL,
)
INDIRECT_REFERENCE_CLAUSE = re.compile(
    r"\b(?:referred to|set out)\s+in\s+"
    r"(?:this\s+)?(?:paragraph|point|article|section|chapter)s?\b",
    re.IGNORECASE,
)
DEFINITION_CLAUSE = re.compile(
    r"^\s*(?:\(\d+[a-z]?\)\s*)?[‘'\"“].+?[’'\"”]\s+means\b",
    re.IGNORECASE | re.DOTALL,
)

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
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "which",
}

DOCUMENT_NAME_STOP_WORDS = {
    "doc",
    "docx",
    "en",
    "eu",
    "md",
    "pdf",
    "txt",
}

NORMATIVE_TERMS = {
    "apply",
    "must",
    "obligation",
    "prohibition",
    "prohibited",
    "required",
    "shall",
}

COMPARISON_META_TERMS = {
    "address",
    "addresses",
    "compare",
    "comparison",
    "differ",
    "difference",
    "differently",
    "regulate",
    "regulates",
    "rules",
}


class QuestionScope(StrEnum):
    DEFINITION = "definition"
    FOCUSED = "focused"
    COMPLETE_LIST = "complete_list"
    CROSS_DOCUMENT = "cross_document"


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
    plan = build_retrieval_plan(question)
    question_type = plan.question_type
    if question_type == QuestionType.DEFINITION:
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
    if re.match(r"^(?:what|which) (?:must|should) .+\bdo\b", normalized):
        return QuestionScope.COMPLETE_LIST
    if top_point is not None:
        return QuestionScope.FOCUSED
    if plan.requires_complete_list:
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


def _query_coverage(question: str, document: str) -> float:
    query_tokens = set(_query_tokens(question))
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(_keyword_tokens(document))) / len(query_tokens)


def _legal_hierarchy_context_scores(question: str, chunks: list[dict]) -> np.ndarray:
    """Score a legal parent clause together with its child points when needed."""
    scores = np.zeros(len(chunks), dtype=np.float32)
    hierarchy_groups: dict[tuple[int, str, str], list[int]] = {}
    for index, chunk in enumerate(chunks):
        if chunk["article"] is None or chunk["paragraph"] is None:
            continue
        key = (chunk["document_id"], chunk["article"], chunk["paragraph"])
        hierarchy_groups.setdefault(key, []).append(index)

    for indices in hierarchy_groups.values():
        context = "\n".join(
            part
            for index in indices
            for part in (chunks[index]["section"], chunks[index]["content"])
            if part
        )
        coverage = _query_coverage(question, context)
        for index in indices:
            scores[index] = coverage
    return scores


def _automatic_expansion_terms(
    question: str,
    documents: list[str],
    similarities: np.ndarray,
    group_keys: list[tuple[int, str | None]],
) -> tuple[str, ...]:
    if len(documents) < MIN_EXPANSION_DOCUMENTS:
        return ()

    query_tokens = set(_query_tokens(question))
    seed_count = min(PSEUDO_RELEVANCE_CHUNKS, len(documents))
    seed_indices = np.argsort(similarities)[::-1][:seed_count]
    anchor_group = group_keys[int(seed_indices[0])]
    coherent_indices = [
        int(index) for index in seed_indices if group_keys[int(index)] == anchor_group
    ]
    if len(coherent_indices) < MIN_EXPANSION_DOCUMENTS:
        return ()

    seed_tokens = [set(_keyword_tokens(documents[index])) for index in coherent_indices]
    seed_frequency = Counter(token for tokens in seed_tokens for token in tokens)
    document_tokens = [set(_keyword_tokens(document)) for document in documents]
    candidates: list[tuple[float, str]] = []

    for token, frequency in seed_frequency.items():
        if token in query_tokens or frequency < MIN_EXPANSION_DOCUMENTS:
            continue

        document_frequency = sum(token in tokens for tokens in document_tokens)
        inverse_document_frequency = math.log(
            1 + len(documents) / max(document_frequency, 1)
        )
        candidates.append((frequency * inverse_document_frequency, token))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return tuple(token for _, token in candidates[:MAX_EXPANSION_TERMS])


def _referenced_clause_target_index(
    question: str,
    chunks: list[dict],
    anchor_index: int,
    ranking_scores: np.ndarray,
) -> int | None:
    anchor = chunks[anchor_index]
    query_tokens = set(_query_tokens(question))
    reference_chunks = [anchor]
    if anchor["article"] is not None and anchor["paragraph"] is None:
        reference_chunks = [
            chunk
            for chunk in chunks
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
        ]

    references: dict[tuple[str, str], tuple[float, int]] = {}
    for chunk in reference_chunks:
        chunk_tokens = set(_keyword_tokens(chunk["content"]))
        coverage = (
            len(query_tokens & chunk_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        for article_number, paragraph in ARTICLE_CITATION.findall(chunk["content"]):
            if str(anchor["article"]).casefold() == f"article {article_number}".casefold():
                continue
            previous_coverage, frequency = references.get(
                (article_number, paragraph),
                (0.0, 0),
            )
            references[(article_number, paragraph)] = (
                max(previous_coverage, coverage),
                frequency + 1,
            )

    ranked_references = sorted(
        references.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True,
    )
    for (article_number, paragraph), (coverage, frequency) in ranked_references:
        if coverage < MIN_QUERY_COVERAGE or frequency < 2 or not paragraph:
            continue

        candidates = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and str(chunk["article"]).casefold() == f"article {article_number}".casefold()
            and (not paragraph or chunk["paragraph"] == paragraph)
            and chunk["point"] is None
            and chunk["subpoint"] is None
        ]
        if candidates:
            return max(candidates, key=lambda index: float(ranking_scores[index]))
    return None


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

    clause_start = r"(?:^|\n)\s*(?:\(\d+[a-z]?\)\s*)?"
    quoted_or_plain_subject = (
        rf"(?:[‘'\"“]{re.escape(subject)}[’'\"”]|{re.escape(subject)})"
    )
    pattern = re.compile(
        rf"{clause_start}{quoted_or_plain_subject}\s+means\b",
        re.IGNORECASE,
    )
    for index, document in enumerate(documents):
        if pattern.search(document):
            scores[index] = 1.0
    return scores


def _question_type_adjustment(question: str, document: str) -> float:
    """Prefer clauses that state the kind of answer requested by the question."""
    question_type = classify_question_type(question)
    normalized = document.casefold()
    if question_type == QuestionType.COMPARISON:
        comparison_requests_definition = bool(
            re.search(r"\b(?:define|definition|meaning)\b", question, re.IGNORECASE)
        )
        if not comparison_requests_definition and DEFINITION_CLAUSE.search(document):
            return -DEFINITION_CLAUSE_PENALTY
        operative_terms = {
            "shall",
            "must",
            "required",
            "prohibited",
            "prohibition",
            "shall not",
        }
        return (
            QUESTION_TYPE_MATCH_BONUS
            if operative_terms & set(_keyword_tokens(normalized))
            else 0.0
        )
    if question_type in {
        QuestionType.OBLIGATION,
        QuestionType.CONTENT,
        QuestionType.CONDITION,
    } and (REFERENCE_CLAUSE.search(document) or INDIRECT_REFERENCE_CLAUSE.search(document)):
        return -REFERENCE_CLAUSE_PENALTY

    matching_terms = {
        QuestionType.OBLIGATION: {"must", "shall", "required", "prohibited"},
        QuestionType.CONTENT: {"contain", "include", "information", "details"},
        QuestionType.CONDITION: {"when", "where", "unless", "within", "before", "after"},
    }.get(question_type, set())
    return QUESTION_TYPE_MATCH_BONUS if matching_terms & set(_keyword_tokens(normalized)) else 0.0


def is_list_question(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    list_terms = {"list", "what are", "which"}
    return any(term in normalized for term in list_terms) or (
        normalized.startswith("what ") and " are " in normalized
    )


def is_comparison_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in COMPARISON_PATTERNS)


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


def phrase_match_scores(question: str, documents: list[str]) -> np.ndarray:
    query_tokens = _query_tokens(question)
    phrases = {
        " ".join(query_tokens[start : start + length])
        for length in range(2, min(4, len(query_tokens)) + 1)
        for start in range(len(query_tokens) - length + 1)
        if not set(query_tokens[start : start + length]) <= COMPARISON_META_TERMS
    }
    scores = np.zeros(len(documents), dtype=np.float32)
    for index, document in enumerate(documents):
        normalized_document = " ".join(_keyword_tokens(document))
        scores[index] = max(
            (len(phrase.split()) for phrase in phrases if phrase in normalized_document),
            default=0,
        )
    maximum = float(scores.max())
    return scores / maximum if maximum > 0 else scores


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
    """Expand a legal clause with the hierarchy needed to read it correctly."""
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

    selected_indices = [
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

    parent_indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["document_id"] == anchor["document_id"]
        and chunk["article"] == anchor["article"]
        and chunk["paragraph"] == anchor["paragraph"]
        and chunk["point"] is None
        and chunk["subpoint"] is None
    ]

    if anchor["subpoint"] is not None:
        point_indices = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
            and chunk["paragraph"] == anchor["paragraph"]
            and chunk["point"] == anchor["point"]
            and chunk["subpoint"] is None
        ]
        return list(dict.fromkeys([*parent_indices, *point_indices, *selected_indices]))

    if anchor["point"] is not None:
        descendants = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
            and chunk["paragraph"] == anchor["paragraph"]
            and chunk["point"] == anchor["point"]
            and chunk["subpoint"] is not None
        ]
    else:
        descendants = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
            and chunk["paragraph"] == anchor["paragraph"]
            and chunk["point"] is not None
        ]

    return list(dict.fromkeys([*parent_indices, *selected_indices, *descendants]))


def _paragraph_anchor_index(
    question: str,
    chunks: list[dict],
    article_anchor_index: int,
    ranking_scores: np.ndarray,
) -> int | None:
    article_anchor = chunks[article_anchor_index]
    query_tokens = set(_query_tokens(question))
    query_contrast = query_tokens & CONTRAST_TERMS
    question_type = classify_question_type(question)
    paragraph_groups: dict[str, list[int]] = {}

    for index, chunk in enumerate(chunks):
        if (
            chunk["document_id"] == article_anchor["document_id"]
            and chunk["article"] == article_anchor["article"]
            and chunk["paragraph"] is not None
        ):
            paragraph_groups.setdefault(chunk["paragraph"], []).append(index)

    candidates: list[tuple[float, int]] = []
    for indices in paragraph_groups.values():
        parent_indices = [
            index
            for index in indices
            if chunks[index]["point"] is None and chunks[index]["subpoint"] is None
        ]
        if not parent_indices:
            continue

        paragraph_tokens = set(
            _keyword_tokens(
                "\n".join(chunks[index]["content"] for index in indices)
            )
        )
        has_nested_points = any(
            chunks[index]["point"] is not None or chunks[index]["subpoint"] is not None
            for index in indices
        )
        coverage = (
            len(query_tokens & paragraph_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        contrast_bonus = (
            PARAGRAPH_CONTRAST_BONUS
            if query_contrast and query_contrast <= paragraph_tokens
            else 0.0
        )
        timing_bonus = (
            PARAGRAPH_TIMING_BONUS
            if question_type == QuestionType.CONDITION
            and paragraph_tokens & TIMING_TEXT_TERMS
            else 0.0
        )
        content_bonus = (
            PARAGRAPH_CONTENT_BONUS
            if question_type == QuestionType.CONTENT
            and (paragraph_tokens & CONTENT_TEXT_TERMS or has_nested_points)
            else 0.0
        )
        condition_bonus = (
            PARAGRAPH_CONDITION_BONUS
            if question_type == QuestionType.CONDITION
            and (paragraph_tokens & CONDITION_TEXT_TERMS or has_nested_points)
            else 0.0
        )
        group_score = (
            max(float(ranking_scores[index]) for index in indices)
            + PARAGRAPH_QUERY_COVERAGE_WEIGHT * coverage
            + contrast_bonus
            + timing_bonus
            + content_bonus
            + condition_bonus
        )
        parent_index = max(
            parent_indices,
            key=lambda index: float(ranking_scores[index]),
        )
        candidates.append((group_score, parent_index))

    if not candidates:
        return None
    return max(candidates)[1]


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
    paragraph_has_points = any(
        chunk["document_id"] == document_id
        and chunk["article"] == article
        and chunk["paragraph"] == original_top["paragraph"]
        and chunk["point"] is not None
        for chunk in chunks
    )
    if (
        original_top["document_id"] == document_id
        and original_top["article"] == article
        and original_top["paragraph"] is not None
        and original_top["point"] is None
        and paragraph_has_points
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


def _dependent_article_indices(
    question: str,
    chunks: list[dict],
    selected_indices: list[int],
    ranking_scores: np.ndarray,
) -> list[int]:
    query_tokens = set(_query_tokens(question))
    requests_clause_contents = bool(
        query_tokens
        & {
            "contain",
            "contains",
            "content",
            "details",
            "include",
            "includes",
            "information",
        }
    )
    selected_articles = {chunks[index]["article"] for index in selected_indices}
    dependencies: list[int] = []

    for source_index in selected_indices:
        source = chunks[source_index]
        source_tokens = set(_keyword_tokens(source["content"]))
        coverage = (
            len(query_tokens & source_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        if coverage < MIN_QUERY_COVERAGE:
            continue
        for article_number, paragraph in ARTICLE_CITATION.findall(source["content"]):
            article = f"article {article_number}".casefold()
            if any(str(value).casefold() == article for value in selected_articles):
                continue
            article_indices = [
                index
                for index, chunk in enumerate(chunks)
                if chunk["document_id"] == source["document_id"]
                and str(chunk["article"]).casefold() == article
                and chunk["point"] is None
                and chunk["subpoint"] is None
            ]
            governing = [
                index for index in article_indices if chunks[index]["paragraph"] == "1"
            ]
            cited = [
                index
                for index in article_indices
                if paragraph and chunks[index]["paragraph"] == paragraph
            ]
            if governing:
                dependencies.append(
                    max(governing, key=lambda index: float(ranking_scores[index]))
                )
            if cited:
                cited_index = max(
                    cited,
                    key=lambda index: float(ranking_scores[index]),
                )
                has_child_points = any(
                    chunk["document_id"] == source["document_id"]
                    and str(chunk["article"]).casefold() == article
                    and chunk["paragraph"] == paragraph
                    and chunk["point"] is not None
                    for chunk in chunks
                )
                if requests_clause_contents or not has_child_points:
                    dependencies.extend(
                        _focused_clause_indices(chunks, cited_index)
                    )
            if dependencies:
                return list(dict.fromkeys(dependencies))
    return []


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


def _merged_retrieved_chunks(
    chunks: list[dict],
    indices: list[int],
    similarities: np.ndarray,
) -> list[RetrievedChunk]:
    grouped: dict[tuple, list[int]] = {}
    for index in indices:
        chunk = chunks[index]
        hierarchy = (
            chunk["article"],
            chunk["paragraph"],
            chunk["point"],
            chunk["subpoint"],
        )
        key = (chunk["document_id"], *hierarchy)
        if not any(hierarchy):
            key = (*key, chunk["id"])
        grouped.setdefault(key, []).append(index)

    results: list[RetrievedChunk] = []
    for grouped_indices in grouped.values():
        base_index = grouped_indices[0]
        base = _retrieved_chunk(
            chunks[base_index],
            max(float(similarities[index]) for index in grouped_indices),
        )
        contents = list(
            dict.fromkeys(chunks[index]["content"].strip() for index in grouped_indices)
        )
        if len(contents) > 1:
            base.content = "\n".join(contents)
        results.append(base)
    return results

def _document_name_tokens(file_name: str) -> set[str]:
    return {
        token
        for token in _keyword_tokens(file_name)
        if token not in DOCUMENT_NAME_STOP_WORDS and not token.isdigit()
    }


def _document_name_match(question: str, file_name: str) -> float:
    query_tokens = set(_query_tokens(question))
    name_tokens = _document_name_tokens(file_name)
    if not name_tokens:
        return 0.0
    return len(query_tokens & name_tokens) / len(name_tokens)


def mentions_multiple_documents(question: str, file_names: list[str]) -> bool:
    return sum(_document_name_match(question, name) > 0 for name in file_names) >= 2


def _explicit_article_number(question: str, file_name: str) -> str | None:
    matches = list(ARTICLE_CITATION.finditer(question))
    if not matches:
        return None

    name_tokens = _document_name_tokens(file_name)
    for match in matches:
        prefix = question[max(0, match.start() - 50) : match.start()]
        prefix_tokens = set(_query_tokens(prefix))
        if prefix_tokens & name_tokens:
            return match.group(1)

    return None


def _article_anchor_index(
    article_number: str,
    chunks: list[dict],
    indices: list[int],
    ranking_scores: np.ndarray,
) -> int | None:
    article = f"article {article_number}".casefold()
    matching = [
        index
        for index in indices
        if str(chunks[index]["article"]).casefold() == article
    ]
    if not matching:
        return None
    headings = [
        index
        for index in matching
        if chunks[index]["paragraph"] is None
        and chunks[index]["point"] is None
        and chunks[index]["subpoint"] is None
    ]
    candidates = headings or matching
    return max(candidates, key=lambda index: float(ranking_scores[index]))


def _explicit_query_anchor(
    question: str,
    chunks: list[dict],
    ranking_scores: np.ndarray,
) -> int | None:
    document_indices: dict[int, list[int]] = {}
    for index, chunk in enumerate(chunks):
        document_indices.setdefault(chunk["document_id"], []).append(index)

    anchors = []
    for indices in document_indices.values():
        article_number = _explicit_article_number(
            question,
            chunks[indices[0]]["original_name"],
        )
        if article_number is None:
            continue
        anchor = _article_anchor_index(
            article_number,
            chunks,
            indices,
            ranking_scores,
        )
        if anchor is not None:
            anchors.append(anchor)
    if anchors:
        return max(anchors, key=lambda index: float(ranking_scores[index]))

    citations = ARTICLE_CITATION.findall(question)
    if len(citations) != 1:
        return None
    article_number, _ = citations[0]
    return _article_anchor_index(
        article_number,
        chunks,
        list(range(len(chunks))),
        ranking_scores,
    )


def _legal_authority_adjustment(chunk: dict) -> float:
    if chunk["article"] is not None:
        return OPERATIVE_CLAUSE_BONUS + _normative_clause_bonus(chunk)
    section = str(chunk.get("section") or "").casefold()
    return -PREAMBLE_PENALTY if "preamble" in section else 0.0


def _normative_clause_bonus(chunk: dict) -> float:
    if chunk["article"] is None:
        return 0.0
    content_tokens = set(
        _keyword_tokens(
            " ".join(
                part for part in (chunk.get("section"), chunk["content"]) if part
            )
        )
    )
    return NORMATIVE_CLAUSE_BONUS if content_tokens & NORMATIVE_TERMS else 0.0


def _external_reference_anchor(
    selected_indices: list[int],
    chunks: list[dict],
    selected_document_ids: set[int],
    ranking_scores: np.ndarray,
) -> tuple[int, int] | None:
    """Resolve an EU Regulation citation into an article in another selected file."""
    for source_index in selected_indices:
        for article_number, year, regulation_number in (
            EXTERNAL_REGULATION_CITATION.findall(chunks[source_index]["content"])
        ):
            for target_document_id in selected_document_ids:
                if target_document_id == chunks[source_index]["document_id"]:
                    continue
                target_indices = [
                    index
                    for index, chunk in enumerate(chunks)
                    if chunk["document_id"] == target_document_id
                ]
                if not target_indices:
                    continue
                file_name_numbers = {
                    str(int(number))
                    for number in re.findall(
                        r"\d+", chunks[target_indices[0]]["original_name"]
                    )
                }
                identifier_matches = (
                    year in file_name_numbers
                    and regulation_number in file_name_numbers
                )
                if not identifier_matches:
                    continue
                article_indices = [
                    index
                    for index in target_indices
                    if str(chunks[index]["article"]).casefold()
                    == f"article {article_number}".casefold()
                ]
                if article_indices:
                    heading_indices = [
                        index
                        for index in article_indices
                        if chunks[index]["paragraph"] is None
                        and chunks[index]["point"] is None
                    ]
                    candidates = heading_indices or article_indices
                    return target_document_id, max(
                        candidates,
                        key=lambda index: float(ranking_scores[index]),
                    )
    return None


def _comparison_clause_indices(
    question: str,
    chunks: list[dict],
    anchor_index: int,
    ranking_scores: np.ndarray,
) -> list[int]:
    anchor = chunks[anchor_index]
    plan = build_retrieval_plan(question)
    maximum_chunks = (
        MAX_COMPARISON_CHUNKS_PER_DOCUMENT
        if plan.requires_complete_list or plan.requires_balanced_sources
        else 1
    )
    document_name_tokens = set().union(
        *(_document_name_tokens(chunk["original_name"]) for chunk in chunks)
    )
    topic_tokens = set(_query_tokens(question)) - document_name_tokens
    topic_tokens -= COMPARISON_META_TERMS
    if anchor["article"] is None:
        same_document = [
            index
            for index in np.argsort(ranking_scores)[::-1]
            if chunks[int(index)]["document_id"] == anchor["document_id"]
        ]
        selected_indices = [int(index) for index in same_document[:maximum_chunks]]
    elif anchor["point"] is not None or anchor["paragraph"] is not None:
        selected_indices = _focused_clause_indices(chunks, anchor_index)
    else:
        paragraph_index = _paragraph_anchor_index(
            question,
            chunks,
            anchor_index,
            ranking_scores,
        )
        selected_indices = (
            _focused_clause_indices(chunks, paragraph_index)
            if paragraph_index is not None
            else [anchor_index]
        )

    broad_rule_question = any(
        term in set(_query_tokens(question))
        for term in {"rule", "rules", "regulate", "regulates"}
    )
    if (
        broad_rule_question
        and not plan.requires_balanced_sources
        and anchor["paragraph"] not in {None, "1"}
        and anchor["point"] is None
    ):
        governing_indices = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
            and chunk["paragraph"] == "1"
            and chunk["point"] is None
            and chunk["subpoint"] is None
        ]
        if governing_indices:
            governing_index = max(
                governing_indices,
                key=lambda index: float(ranking_scores[index]),
            )
            selected_indices = list(
                dict.fromkeys([governing_index, *selected_indices])
            )

    selected_articles = {
        chunks[index]["article"]
        for index in selected_indices
        if chunks[index]["article"]
    }
    ranked_document_indices = [
        int(index)
        for index in np.argsort(ranking_scores)[::-1]
        if chunks[int(index)]["document_id"] == anchor["document_id"]
    ]
    for index in ranked_document_indices:
        if len(selected_indices) >= maximum_chunks:
            break
        article = chunks[index]["article"]
        if (
            index in selected_indices
            or article is None
            or article in selected_articles
            or _normative_clause_bonus(chunks[index]) == 0
        ):
            continue
        candidate_tokens = set(
            _keyword_tokens(
                "\n".join(
                    part
                    for part in (chunks[index]["section"], chunks[index]["content"])
                    if part
                )
            )
        )
        coverage = (
            len(topic_tokens & candidate_tokens) / len(topic_tokens)
            if topic_tokens
            else 0.0
        )
        if coverage < MIN_SUPPLEMENTAL_QUERY_COVERAGE:
            continue
        selected_indices.extend(_focused_clause_indices(chunks, index))
        selected_articles.add(article)

    selected_indices = list(dict.fromkeys(selected_indices))
    return sorted(
        selected_indices,
        key=lambda index: (
            chunks[index]["document_id"],
            chunks[index].get("chunk_index", index),
        ),
    )


def _comparison_indices(
    question: str,
    chunks: list[dict],
    searchable_documents: list[str],
    similarities: np.ndarray,
    ranking_scores: np.ndarray,
    min_similarity: float,
) -> list[int]:
    document_indices: dict[int, list[int]] = {}
    for index, chunk in enumerate(chunks):
        document_indices.setdefault(chunk["document_id"], []).append(index)

    candidates: list[tuple[bool, float, int]] = []
    for indices in document_indices.values():
        file_name = chunks[indices[0]]["original_name"]
        explicit_article = _explicit_article_number(question, file_name)
        anchor_index = (
            _article_anchor_index(
                explicit_article,
                chunks,
                indices,
                ranking_scores,
            )
            if explicit_article is not None
            else None
        )
        if anchor_index is None:
            anchor_index = max(
                indices,
                key=lambda index: (
                    float(ranking_scores[index])
                    + _legal_authority_adjustment(chunks[index])
                ),
            )
        elif not any(paragraph for _, paragraph in ARTICLE_CITATION.findall(question)):
            governing_indices = [
                index
                for index in indices
                if chunks[index]["article"] == chunks[anchor_index]["article"]
                and chunks[index]["paragraph"] == "1"
                and chunks[index]["point"] is None
                and chunks[index]["subpoint"] is None
            ]
            if governing_indices:
                anchor_index = max(
                    governing_indices,
                    key=lambda index: float(ranking_scores[index]),
                )
        anchor = chunks[anchor_index]
        name_match = _document_name_match(question, anchor["original_name"])
        ranking_score = float(ranking_scores[anchor_index])
        relevant = _passes_relevance_gate(
            question,
            searchable_documents[anchor_index],
            float(similarities[anchor_index]),
            0.0,
        )
        if (
            explicit_article is None and ranking_score < min_similarity
        ) or (name_match == 0 and not relevant):
            continue
        candidates.append(
            (
                name_match > 0,
                ranking_score + DOCUMENT_NAME_MATCH_BONUS * name_match,
                anchor_index,
            )
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    mentioned = [candidate for candidate in candidates if candidate[0]]
    if not is_comparison_question(question) and len(mentioned) < 2:
        return []
    selected = mentioned if len(mentioned) >= 2 else candidates
    selected = selected[:MAX_COMPARISON_DOCUMENTS]
    if len(selected) < 2:
        return []

    anchors = {
        chunks[anchor_index]["document_id"]: anchor_index
        for *_, anchor_index in selected
    }
    selected_document_ids = set(anchors)
    for anchor_index in tuple(anchors.values()):
        source_indices = _comparison_clause_indices(
            question,
            chunks,
            anchor_index,
            ranking_scores,
        )
        external_reference = _external_reference_anchor(
            source_indices,
            chunks,
            selected_document_ids,
            ranking_scores,
        )
        if external_reference is not None:
            target_document_id, target_anchor_index = external_reference
            anchors[target_document_id] = target_anchor_index

    comparison_indices: list[int] = []
    for anchor_index in anchors.values():
        clause_indices = _comparison_clause_indices(
            question,
            chunks,
            anchor_index,
            ranking_scores,
        )
        comparison_indices.extend(clause_indices)
    return list(dict.fromkeys(comparison_indices))


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
    searchable_documents = [
        "\n".join(part for part in (chunk["section"], chunk["content"]) if part)
        for chunk in chunks
    ]
    keyword_scores = bm25_scores(
        question,
        searchable_documents,
    )
    maximum_keyword_score = float(keyword_scores.max())
    if maximum_keyword_score > 0:
        keyword_scores /= maximum_keyword_score
    phrase_scores = phrase_match_scores(question, searchable_documents)

    expansion_terms = _automatic_expansion_terms(
        question,
        searchable_documents,
        similarities,
        [
            (chunk["document_id"], chunk["article"] or chunk["section"])
            for chunk in chunks
        ],
    )
    expanded_question = " ".join((question, *expansion_terms))
    expansion_scores = bm25_scores(expanded_question, searchable_documents)
    maximum_expansion_score = float(expansion_scores.max())
    if maximum_expansion_score > 0:
        expansion_scores /= maximum_expansion_score

    definition_scores = _definition_scores(
        question,
        [chunk["content"] for chunk in chunks],
    )
    hierarchy_context_scores = _legal_hierarchy_context_scores(question, chunks)
    base_ranking_scores = (
        similarities
        + KEYWORD_WEIGHT * keyword_scores
        + PHRASE_MATCH_WEIGHT * phrase_scores
        + QUERY_EXPANSION_WEIGHT * expansion_scores
        + DEFINITION_MATCH_BONUS * definition_scores
        + LEGAL_HIERARCHY_CONTEXT_WEIGHT * hierarchy_context_scores
    )
    base_ranking_scores += np.asarray(
        [_question_type_adjustment(question, document) for document in searchable_documents],
        dtype=np.float32,
    )
    ranked_indices = np.argsort(base_ranking_scores)[::-1]
    best_score = float(base_ranking_scores[ranked_indices[0]])
    original_top_index = int(ranked_indices[0])
    explicit_query_anchor = _explicit_query_anchor(
        question,
        chunks,
        base_ranking_scores,
    )
    lexical_scores = keyword_scores + phrase_scores
    lexical_top_index = int(np.argmax(lexical_scores))
    lexical_document = searchable_documents[lexical_top_index]
    strong_lexical_anchor = (
        max(
            float(keyword_scores[lexical_top_index]),
            float(phrase_scores[lexical_top_index]),
        )
        >= LEXICAL_ANCHOR_MIN_SCORE
        and _query_coverage(question, lexical_document)
        >= LEXICAL_ANCHOR_MIN_COVERAGE
    )
    if best_score < min_similarity and not strong_lexical_anchor:
        return []
    if explicit_query_anchor is not None:
        original_top_index = explicit_query_anchor
    elif strong_lexical_anchor:
        original_top_index = lexical_top_index

    semantic_top_index = int(np.argmax(similarities))
    if (
        explicit_query_anchor is None
        and not strong_lexical_anchor
        and _question_type_adjustment(question, searchable_documents[semantic_top_index])
        >= 0
        and float(hierarchy_context_scores[original_top_index])
        < HIERARCHY_CONTEXT_ANCHOR_MIN_COVERAGE
        and float(similarities[semantic_top_index])
        - float(similarities[original_top_index])
        >= SEMANTIC_OVERRIDE_MARGIN
    ):
        original_top_index = semantic_top_index

    comparison_ranking_scores = (
        similarities
        + COMPARISON_KEYWORD_WEIGHT * keyword_scores
        + PHRASE_MATCH_WEIGHT * phrase_scores
        + DEFINITION_MATCH_BONUS * definition_scores
        + LEGAL_HIERARCHY_CONTEXT_WEIGHT * hierarchy_context_scores
    )
    comparison_ranking_scores += np.asarray(
        [_question_type_adjustment(question, document) for document in searchable_documents],
        dtype=np.float32,
    )
    comparison_indices = _comparison_indices(
        question,
        chunks,
        searchable_documents,
        similarities,
        comparison_ranking_scores,
        min_similarity,
    )
    if comparison_indices:
        return _merged_retrieved_chunks(chunks, comparison_indices, similarities)

    reference_target_index = _referenced_clause_target_index(
        expanded_question,
        chunks,
        original_top_index,
        base_ranking_scores,
    )
    if reference_target_index is not None:
        original_top_index = reference_target_index
    top_document = "\n".join(
        part
        for part in (
            chunks[original_top_index]["section"],
            chunks[original_top_index]["content"],
        )
        if part
    )
    if not _passes_relevance_gate(
        expanded_question,
        top_document,
        float(similarities[original_top_index]),
        float(definition_scores[original_top_index]),
    ) and not (
        strong_lexical_anchor
        or float(hierarchy_context_scores[original_top_index])
        >= HIERARCHY_CONTEXT_ANCHOR_MIN_COVERAGE
    ):
        return []

    if reference_target_index is not None:
        return [
            _retrieved_chunk(chunks[index], float(similarities[index]))
            for index in _focused_clause_indices(chunks, original_top_index)
        ]

    scope = classify_question_scope(question, chunks[original_top_index]["point"])
    if scope == QuestionScope.DEFINITION:
        selected_indices = _definition_indices(definition_scores, base_ranking_scores)
        if not np.any(definition_scores > 0):
            selected_indices = _focused_clause_indices(
                chunks,
                selected_indices[0],
            )
        return [
            _retrieved_chunk(chunks[index], float(similarities[index]))
            for index in selected_indices
        ]

    if scope == QuestionScope.COMPLETE_LIST:
        if chunks[original_top_index]["article"] is not None:
            anchor_index = original_top_index
        else:
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
            dependent_indices = _dependent_article_indices(
                question,
                chunks,
                selected_indices,
                base_ranking_scores,
            )
            if dependent_indices:
                parent_indices = [
                    index
                    for index in selected_indices
                    if chunks[index]["point"] is None
                    and chunks[index]["subpoint"] is None
                ][:2]
                selected_indices = list(
                    dict.fromkeys([*dependent_indices, *parent_indices])
                )
            return [
                _retrieved_chunk(chunks[index], float(similarities[index]))
                for index in selected_indices
            ]

    if chunks[original_top_index]["article"] is not None:
        if chunks[original_top_index]["point"] is None:
            paragraph_anchor_index = _paragraph_anchor_index(
                expanded_question,
                chunks,
                original_top_index,
                base_ranking_scores,
            )
            if paragraph_anchor_index is None:
                selected_indices = _complete_scope_indices(
                    chunks,
                    original_top_index,
                    original_top_index,
                )
            else:
                selected_indices = _focused_clause_indices(
                    chunks,
                    paragraph_anchor_index,
                )
        else:
            selected_indices = _focused_clause_indices(chunks, original_top_index)
        return [
            _retrieved_chunk(chunks[index], float(similarities[index]))
            for index in selected_indices
        ]

    if strong_lexical_anchor:
        ranked_indices = np.asarray(
            [
                original_top_index,
                *(
                    int(index)
                    for index in ranked_indices
                    if int(index) != original_top_index
                    and chunks[int(index)]["document_id"]
                    == chunks[original_top_index]["document_id"]
                ),
            ],
            dtype=int,
        )
    accepted_score = max(
        min_similarity,
        float(base_ranking_scores[original_top_index]) - max_score_drop,
    )
    plan = build_retrieval_plan(question)
    list_question = plan.requires_complete_list
    section_context_question = plan.prefers_section_context
    if list_question or section_context_question:
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
    if top_chunk["section"] or section_context_question:
        for index, chunk in enumerate(chunks):
            if (
                index == top_index
                or chunk["document_id"] != top_chunk["document_id"]
            ):
                continue

            distance = abs(chunk["chunk_index"] - top_chunk["chunk_index"])
            same_section = top_chunk["section"] and chunk["section"] == top_chunk["section"]
            nearby_unsectioned_context = (
                top_chunk["section"] is None
                and section_context_question
                and distance <= UNSECTIONED_CONTEXT_WINDOW
            )
            if same_section and (distance == 1 or list_question or section_context_question):
                ranking_scores[index] += ADJACENT_CHUNK_BONUS / distance
                related_indices.add(index)
            elif nearby_unsectioned_context:
                ranking_scores[index] += ADJACENT_CHUNK_BONUS / distance
                related_indices.add(index)

        remaining_indices = [
            int(index) for index in ranked_indices if int(index) != top_index
        ]
        remaining_indices.sort(
            key=lambda index: float(ranking_scores[index]),
            reverse=True,
        )
        if list_question or section_context_question:
            section_indices = sorted(
                {top_index, *related_indices},
                key=lambda index: chunks[index]["chunk_index"],
            )
            remaining_indices = [
                index for index in remaining_indices if index not in section_indices
            ]
            ranked_indices = np.asarray(
                [*section_indices, *remaining_indices],
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
            and not (
                (list_question or section_context_question)
                and int(index) in related_indices
            )
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
