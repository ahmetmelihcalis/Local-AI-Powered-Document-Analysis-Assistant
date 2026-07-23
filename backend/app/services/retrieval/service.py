import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from app.database import DATABASE_PATH
from app.repositories.document_repository import get_chunks
from app.services.foundry_service import create_embeddings
from . import expansion, scoring
from .questions import (
    QuestionType,
    build_retrieval_plan,
    enrich_question,
)

# Internal orchestration helpers share the same tokenisation rules as scoring.
_keyword_tokens = scoring.keyword_tokens
_query_tokens = scoring.query_tokens


DEFAULT_TOP_K = 4
MIN_SIMILARITY = 0.30
MAX_SCORE_DROP = 0.10
KEYWORD_WEIGHT = 0.12
QUERY_EXPANSION_WEIGHT = 0.08
COMPARISON_KEYWORD_WEIGHT = 0.25
PHRASE_MATCH_WEIGHT = 0.20
DEFINITION_MATCH_BONUS = 0.15
HIGH_CONFIDENCE_SCORE = 0.50
MEDIUM_CONFIDENCE_SCORE = 0.35
MIN_RESULT_COUNT = 2
ADJACENT_CHUNK_BONUS = 0.10
UNSECTIONED_CONTEXT_WINDOW = 2
LEGAL_HIERARCHY_CONTEXT_WEIGHT = 0.20
HIERARCHY_CONTEXT_ANCHOR_MIN_COVERAGE = 0.60
SEMANTIC_OVERRIDE_MARGIN = 0.06
LEXICAL_ANCHOR_MIN_SCORE = 0.75
LEXICAL_ANCHOR_MIN_COVERAGE = 0.60
MAX_COMPARISON_DOCUMENTS = 3
MAX_COMPARISON_CHUNKS_PER_DOCUMENT = 2
MIN_SUPPLEMENTAL_QUERY_COVERAGE = 0.60

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
    if top_point is not None and re.search(
        r"\barticle\s+\d+[a-z]?\s*\(\d+\)\s*\([a-z]\)",
        normalized,
    ):
        return QuestionScope.FOCUSED
    if plan.requires_complete_list:
        return QuestionScope.COMPLETE_LIST
    if top_point is not None:
        return QuestionScope.FOCUSED
    return QuestionScope.FOCUSED


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
        coverage = scoring.query_coverage(question, context)
        for index in indices:
            scores[index] = coverage
    return scores


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


def is_comparison_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in COMPARISON_PATTERNS)


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
        *(scoring.document_name_tokens(chunk["original_name"]) for chunk in chunks)
    )
    topic_tokens = set(_query_tokens(question)) - document_name_tokens
    topic_tokens -= scoring.COMPARISON_META_TERMS
    if anchor["article"] is None:
        same_document = [
            index
            for index in np.argsort(ranking_scores)[::-1]
            if chunks[int(index)]["document_id"] == anchor["document_id"]
        ]
        selected_indices = [int(index) for index in same_document[:maximum_chunks]]
    elif anchor["point"] is not None or anchor["paragraph"] is not None:
        selected_indices = expansion.focused_clause_indices(chunks, anchor_index)
    else:
        paragraph_index = expansion.paragraph_anchor_index(
            question,
            chunks,
            anchor_index,
            ranking_scores,
        )
        selected_indices = (
            expansion.focused_clause_indices(chunks, paragraph_index)
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
            or scoring.normative_clause_bonus(chunks[index]) == 0
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
        selected_indices.extend(expansion.focused_clause_indices(chunks, index))
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
    selected_anchor_indices = scoring.comparison_anchor_indices(
        question,
        chunks,
        searchable_documents,
        similarities,
        ranking_scores,
        min_similarity,
        MAX_COMPARISON_DOCUMENTS,
    )
    if selected_anchor_indices:
        anchors = {
            chunks[anchor_index]["document_id"]: anchor_index
            for anchor_index in selected_anchor_indices
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
            comparison_indices.extend(
                _comparison_clause_indices(
                    question,
                    chunks,
                    anchor_index,
                    ranking_scores,
                )
            )
        return list(dict.fromkeys(comparison_indices))

    return []


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

    question = enrich_question(question)

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

    similarities = scoring.cosine_similarities(
        response[0],
        [chunk["embedding"] for chunk in chunks],
    )
    searchable_documents = [
        "\n".join(part for part in (chunk["section"], chunk["content"]) if part)
        for chunk in chunks
    ]
    keyword_scores = scoring.bm25_scores(
        question,
        searchable_documents,
    )
    maximum_keyword_score = float(keyword_scores.max())
    if maximum_keyword_score > 0:
        keyword_scores /= maximum_keyword_score
    phrase_scores = scoring.phrase_match_scores(question, searchable_documents)

    expansion_terms = scoring.automatic_expansion_terms(
        question,
        searchable_documents,
        similarities,
        [
            (chunk["document_id"], chunk["article"] or chunk["section"])
            for chunk in chunks
        ],
    )
    expanded_question = " ".join((question, *expansion_terms))
    expansion_scores = scoring.bm25_scores(expanded_question, searchable_documents)
    maximum_expansion_score = float(expansion_scores.max())
    if maximum_expansion_score > 0:
        expansion_scores /= maximum_expansion_score

    definition_scores = scoring.definition_scores(
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
        [scoring.question_type_adjustment(question, document) for document in searchable_documents],
        dtype=np.float32,
    )
    ranked_indices = np.argsort(base_ranking_scores)[::-1]
    best_score = float(base_ranking_scores[ranked_indices[0]])
    original_top_index = int(ranked_indices[0])
    explicit_query_anchor = scoring.explicit_query_anchor(
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
        and scoring.query_coverage(question, lexical_document)
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
        and scoring.question_type_adjustment(question, searchable_documents[semantic_top_index])
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
        [scoring.question_type_adjustment(question, document) for document in searchable_documents],
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
    if not scoring.passes_relevance_gate(
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
            for index in expansion.focused_clause_indices(chunks, original_top_index)
        ]

    scope = classify_question_scope(question, chunks[original_top_index]["point"])
    if scope == QuestionScope.DEFINITION:
        selected_indices = _definition_indices(definition_scores, base_ranking_scores)
        if not np.any(definition_scores > 0):
            selected_indices = expansion.focused_clause_indices(
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
            content_paragraph_anchor = (
                expansion.paragraph_anchor_index(
                    question,
                    chunks,
                    anchor_index,
                    base_ranking_scores,
                )
                if build_retrieval_plan(question).question_type == QuestionType.CONTENT
                else None
            )
            selected_indices = expansion.complete_scope_indices(
                chunks,
                content_paragraph_anchor or anchor_index,
                content_paragraph_anchor or original_top_index,
            )
            dependent_indices = expansion.dependent_article_indices(
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
            paragraph_anchor_index = expansion.paragraph_anchor_index(
                expanded_question,
                chunks,
                original_top_index,
                base_ranking_scores,
            )
            if paragraph_anchor_index is None:
                selected_indices = expansion.complete_scope_indices(
                    chunks,
                    original_top_index,
                    original_top_index,
                )
            else:
                selected_indices = expansion.focused_clause_indices(
                    chunks,
                    paragraph_anchor_index,
                )
        else:
            selected_indices = expansion.focused_clause_indices(chunks, original_top_index)
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
