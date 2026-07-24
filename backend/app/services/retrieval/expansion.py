import numpy as np

from .questions import QuestionType, classify_question_type
from .scoring import (
    ARTICLE_CITATION,
    MIN_QUERY_COVERAGE,
    keyword_tokens,
    query_tokens,
)


CONTRAST_TERMS = {
    "except",
    "exception",
    "exempt",
    "not",
    "unless",
    "without",
}
TIMING_TEXT_TERMS = {
    "when",
    "within",
    "delay",
    "hours",
    "days",
    "before",
    "after",
}
CONTENT_TEXT_TERMS = {
    "contain",
    "contains",
    "include",
    "includes",
    "information",
    "details",
}
CONDITION_TEXT_TERMS = {"condition", "conditions", "where", "unless", "if"}
MAX_COMPLETE_SCOPE_CHUNKS = 20
PARAGRAPH_QUERY_COVERAGE_WEIGHT = 0.12
PARAGRAPH_CONTRAST_BONUS = 0.12
PARAGRAPH_TIMING_BONUS = 0.16
PARAGRAPH_CONTENT_BONUS = 0.18
PARAGRAPH_CONDITION_BONUS = 0.16


def focused_clause_indices(chunks: list[dict], anchor_index: int) -> list[int]:
    anchor = chunks[anchor_index]
    if anchor["article"] is None:
        return [anchor_index]

    hierarchy_keys = ("document_id", "article", "paragraph", "point", "subpoint")
    hierarchy = tuple(anchor[key] for key in hierarchy_keys)
    selected = [
        index
        for index, chunk in enumerate(chunks)
        if tuple(chunk[key] for key in hierarchy_keys) == hierarchy
    ]
    parents = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["document_id"] == anchor["document_id"]
        and chunk["article"] == anchor["article"]
        and chunk["paragraph"] == anchor["paragraph"]
        and chunk["point"] is None
        and chunk["subpoint"] is None
    ]
    if anchor["subpoint"] is not None:
        points = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["document_id"] == anchor["document_id"]
            and chunk["article"] == anchor["article"]
            and chunk["paragraph"] == anchor["paragraph"]
            and chunk["point"] == anchor["point"]
            and chunk["subpoint"] is None
        ]
        return list(dict.fromkeys([*parents, *points, *selected]))

    descendants = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["document_id"] == anchor["document_id"]
        and chunk["article"] == anchor["article"]
        and chunk["paragraph"] == anchor["paragraph"]
        and (
            (
                anchor["point"] is not None
                and chunk["point"] == anchor["point"]
                and chunk["subpoint"] is not None
            )
            or (anchor["point"] is None and chunk["point"] is not None)
        )
    ]
    return list(dict.fromkeys([*parents, *selected, *descendants]))


def paragraph_anchor_index(
    question: str,
    chunks: list[dict],
    article_anchor_index: int,
    ranking_scores: np.ndarray,
) -> int | None:
    article_anchor = chunks[article_anchor_index]
    query = set(query_tokens(question))
    contrast = query & CONTRAST_TERMS
    question_type = classify_question_type(question)
    groups: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        if (
            chunk["document_id"] == article_anchor["document_id"]
            and chunk["article"] == article_anchor["article"]
            and chunk["paragraph"] is not None
        ):
            groups.setdefault(chunk["paragraph"], []).append(index)

    candidates: list[tuple[float, int]] = []
    for indices in groups.values():
        parents = [
            index
            for index in indices
            if chunks[index]["point"] is None
            and chunks[index]["subpoint"] is None
        ]
        if not parents:
            continue

        content = "\n".join(chunks[index]["content"] for index in indices)
        tokens = set(keyword_tokens(content))
        nested = any(
            chunks[index]["point"] is not None
            or chunks[index]["subpoint"] is not None
            for index in indices
        )
        coverage = len(query & tokens) / len(query) if query else 0.0

        bonus = (
            PARAGRAPH_CONTRAST_BONUS
            if contrast and contrast <= tokens
            else 0.0
        )
        if question_type == QuestionType.CONDITION and tokens & TIMING_TEXT_TERMS:
            bonus += PARAGRAPH_TIMING_BONUS
        if question_type == QuestionType.CONTENT and (tokens & CONTENT_TEXT_TERMS or nested):
            bonus += PARAGRAPH_CONTENT_BONUS
        if question_type == QuestionType.CONDITION and (tokens & CONDITION_TEXT_TERMS or nested):
            bonus += PARAGRAPH_CONDITION_BONUS

        ranking_score = max(float(ranking_scores[index]) for index in indices)
        score = ranking_score + PARAGRAPH_QUERY_COVERAGE_WEIGHT * coverage + bonus
        parent = max(parents, key=lambda index: float(ranking_scores[index]))
        candidates.append((score, parent))

    return max(candidates)[1] if candidates else None


def complete_scope_indices(
    chunks: list[dict],
    anchor_index: int,
    original_top_index: int,
) -> list[int]:
    anchor = chunks[anchor_index]
    original_top = chunks[original_top_index]
    paragraph = None
    has_points = any(
        chunk["document_id"] == anchor["document_id"]
        and chunk["article"] == anchor["article"]
        and chunk["paragraph"] == original_top["paragraph"]
        and chunk["point"] is not None
        for chunk in chunks
    )
    top_is_structured_clause = (
        original_top["document_id"] == anchor["document_id"]
        and original_top["article"] == anchor["article"]
        and original_top["paragraph"] is not None
    )
    if top_is_structured_clause and has_points:
        paragraph = original_top["paragraph"]

    indices = [
        index
        for index, chunk in enumerate(chunks)
        if chunk["document_id"] == anchor["document_id"]
        and chunk["article"] == anchor["article"]
        and (paragraph is None or chunk["paragraph"] == paragraph)
    ]
    substantive = [
        index
        for index in indices
        if chunks[index]["paragraph"] is not None
        or chunks[index]["point"] is not None
    ]
    return (substantive or indices)[:MAX_COMPLETE_SCOPE_CHUNKS]


def dependent_article_indices(
    question: str,
    chunks: list[dict],
    selected_indices: list[int],
    ranking_scores: np.ndarray,
) -> list[int]:
    query = set(query_tokens(question))
    content_terms = {
        "contain",
        "contains",
        "content",
        "details",
        "include",
        "includes",
        "information",
    }
    wants_contents = bool(query & content_terms)
    selected_articles = {chunks[index]["article"] for index in selected_indices}
    dependencies: list[int] = []
    for source_index in selected_indices:
        source = chunks[source_index]
        tokens = set(keyword_tokens(source["content"]))
        coverage = len(query & tokens) / len(query) if query else 0.0
        if coverage < MIN_QUERY_COVERAGE:
            continue
        for article_number, paragraph in ARTICLE_CITATION.findall(source["content"]):
            article = f"article {article_number}".casefold()
            if any(str(value).casefold() == article for value in selected_articles):
                continue
            matches = [
                index for index, chunk in enumerate(chunks)
                if chunk["document_id"] == source["document_id"]
                and str(chunk["article"]).casefold() == article
                and chunk["point"] is None
                and chunk["subpoint"] is None
            ]
            governing = [
                index for index in matches if chunks[index]["paragraph"] == "1"
            ]
            cited = [
                index
                for index in matches
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
                has_children = any(
                    chunk["document_id"] == source["document_id"]
                    and str(chunk["article"]).casefold() == article
                    and chunk["paragraph"] == paragraph
                    and chunk["point"] is not None
                    for chunk in chunks
                )
                if wants_contents or not has_children:
                    dependencies.extend(focused_clause_indices(chunks, cited_index))
            if dependencies:
                return list(dict.fromkeys(dependencies))
    return []
