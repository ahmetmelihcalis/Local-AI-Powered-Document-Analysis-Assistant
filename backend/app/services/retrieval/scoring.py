import math
import re
from collections import Counter

import numpy as np

from .questions import QuestionType, classify_question_type


BM25_K1 = 1.5
BM25_B = 0.75
MIN_RELEVANCE_COSINE = 0.55
HIGH_RELEVANCE_COSINE = 0.68
MIN_QUERY_COVERAGE = 0.30
MIN_EXPANSION_DOCUMENTS = 2
PSEUDO_RELEVANCE_CHUNKS = 4
MAX_EXPANSION_TERMS = 8
QUESTION_TYPE_MATCH_BONUS = 0.10
REFERENCE_CLAUSE_PENALTY = 0.12
DEFINITION_CLAUSE_PENALTY = 0.10
CONDITION_CLAUSE_BONUS = 0.40
DIRECT_CONDITION_OBLIGATION_BONUS = 0.20
ENFORCEMENT_CLAUSE_PENALTY = 0.40
GUIDANCE_CLAUSE_PENALTY = 0.30
DATA_SUBJECT_COMMUNICATION_BONUS = 0.28
SUPERVISORY_NOTIFICATION_PENALTY = 0.20
EXPLICIT_TIMEFRAME_BONUS = 0.42
MISSING_TIMEFRAME_PENALTY = 0.12
ACTOR_ACTION_TARGET_BONUS = 0.55
REVERSED_ACTOR_ACTION_TARGET_PENALTY = 0.50
MAX_ACTOR_ACTION_DISTANCE = 180

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
DOCUMENT_NAME_STOP_WORDS = {"doc", "docx", "en", "eu", "md", "pdf", "txt"}
NORMATIVE_TERMS = {
    "apply",
    "must",
    "obligation",
    "prohibition",
    "prohibited",
    "required",
    "shall",
}
DOCUMENT_NAME_MATCH_BONUS = 0.18
NORMATIVE_CLAUSE_BONUS = 0.05
OPERATIVE_CLAUSE_BONUS = 0.15
PREAMBLE_PENALTY = 0.12
ARTICLE_CITATION = re.compile(r"\bArticle\s+(\d+[a-z]?)(?:\s*\(([0-9]+)\))?", re.IGNORECASE)
COMPARISON_PATTERNS = (
    re.compile(r"\bcompar(?:e|ed|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bdiffer(?:s|ed|ent|ently|ence|ences)?\b", re.IGNORECASE),
    re.compile(r"\b(?:versus|vs\.?)\b", re.IGNORECASE),
)
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
DEFINITION_REQUEST = re.compile(r"\b(?:define|definition|meaning)\b", re.IGNORECASE)
CONTROLLER_NOTIFIES_SUPERVISORY_AUTHORITY = re.compile(
    rf"\bcontroller\b.{{0,{MAX_ACTOR_ACTION_DISTANCE}}}\bnotif(?:y|ies|ied|ication)\b"
    rf".{{0,{MAX_ACTOR_ACTION_DISTANCE}}}"
    r"\bsupervisory\s+authority\b",
    re.IGNORECASE,
)
SUPERVISORY_AUTHORITY_NOTIFIES_CONTROLLER = re.compile(
    rf"\bsupervisory\s+authority\b.{{0,{MAX_ACTOR_ACTION_DISTANCE}}}"
    rf"\bnotif(?:y|ies|ied|ication)\b.{{0,{MAX_ACTOR_ACTION_DISTANCE}}}\bcontroller\b",
    re.IGNORECASE,
)

DIRECT_RULE_TERMS = {"shall", "must", "required"}
CONDITION_SIGNALS = {"unless", "likely", "where", "within", "delay", "deadline"}
PERSON_TERMS = {"person", "people", "individual", "subject", "user"}
INFORM_TERMS = {"tell", "inform", "communicate"}
DATA_SUBJECT_COMMUNICATION_TERMS = {"data", "subject", "communicate"}
SUPERVISORY_NOTIFICATION_TERMS = {"supervisory", "authority", "notify"}
TIMEFRAME_TERMS = {"hour", "hours", "day", "days"}


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
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    similarities = np.full(matrix.shape[0], -1.0, dtype=np.float32)
    np.divide(matrix @ query, denominators, out=similarities, where=denominators > 0)
    return similarities


def keyword_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in KEYWORD_STOP_WORDS
    ]


def query_tokens(question: str) -> list[str]:
    return list(dict.fromkeys(keyword_tokens(question)))


def query_coverage(question: str, document: str) -> float:
    tokens = set(query_tokens(question))
    if not tokens:
        return 0.0
    return len(tokens & set(keyword_tokens(document))) / len(tokens)


def automatic_expansion_terms(
    question: str,
    documents: list[str],
    similarities: np.ndarray,
    group_keys: list[tuple[int, str | None]],
) -> tuple[str, ...]:
    if len(documents) < MIN_EXPANSION_DOCUMENTS:
        return ()
    query = set(query_tokens(question))
    seeds = np.argsort(similarities)[::-1][
        : min(PSEUDO_RELEVANCE_CHUNKS, len(documents))
    ]
    anchor_group = group_keys[int(seeds[0])]
    coherent = [int(index) for index in seeds if group_keys[int(index)] == anchor_group]
    if len(coherent) < MIN_EXPANSION_DOCUMENTS:
        return ()
    frequencies = Counter(
        token
        for index in coherent
        for token in set(keyword_tokens(documents[index]))
    )
    document_tokens = [set(keyword_tokens(document)) for document in documents]
    candidates = []
    for token, frequency in frequencies.items():
        if token in query or frequency < MIN_EXPANSION_DOCUMENTS:
            continue
        document_frequency = sum(token in tokens for tokens in document_tokens)
        expansion_weight = frequency * math.log(
            1 + len(documents) / max(document_frequency, 1)
        )
        candidates.append((expansion_weight, token))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return tuple(token for _, token in candidates[:MAX_EXPANSION_TERMS])


def passes_relevance_gate(
    question: str,
    document: str,
    cosine_score: float,
    definition_score: float,
) -> bool:
    if definition_score > 0 or cosine_score >= HIGH_RELEVANCE_COSINE:
        return True
    return (
        cosine_score >= MIN_RELEVANCE_COSINE
        and query_coverage(question, document) >= MIN_QUERY_COVERAGE
    )


def definition_scores(question: str, documents: list[str]) -> np.ndarray:
    normalized = " ".join(question.casefold().strip(" ?.!\t\n").split())
    if not normalized.startswith("what is "):
        return np.zeros(len(documents), dtype=np.float32)
    subject = normalized.removeprefix("what is ")
    for article in ("a ", "an ", "the "):
        if subject.startswith(article):
            subject = subject.removeprefix(article)
            break
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\(\d+[a-z]?\)\s*)?"
        rf"(?:[‘'\"“]{re.escape(subject)}[’'\"”]|{re.escape(subject)})\s+means\b",
        re.IGNORECASE,
    )
    return np.asarray(
        [1.0 if pattern.search(document) else 0.0 for document in documents],
        dtype=np.float32,
    )


def question_type_adjustment(question: str, document: str) -> float:
    question_type = classify_question_type(question)
    if question_type == QuestionType.COMPARISON:
        return _comparison_adjustment(question, document)

    tokens = set(keyword_tokens(document))
    question_types_with_direct_rules = {
        QuestionType.OBLIGATION,
        QuestionType.CONTENT,
        QuestionType.CONDITION,
    }
    is_indirect_reference = (
        REFERENCE_CLAUSE.search(document)
        or INDIRECT_REFERENCE_CLAUSE.search(document)
    )
    if question_type in question_types_with_direct_rules and is_indirect_reference:
        return -REFERENCE_CLAUSE_PENALTY

    matching_terms = {
        QuestionType.OBLIGATION: {"must", "shall", "required", "prohibited"},
        QuestionType.CONTENT: {"contain", "include", "information", "details"},
        QuestionType.CONDITION: {
            "when",
            "where",
            "unless",
            "within",
            "before",
            "after",
        },
    }.get(question_type, set())
    adjustment = QUESTION_TYPE_MATCH_BONUS if matching_terms & tokens else 0.0
    if question_type == QuestionType.CONDITION:
        adjustment += _condition_adjustment(question, document, tokens)
    return adjustment


def _comparison_adjustment(question: str, document: str) -> float:
    requests_definition = DEFINITION_REQUEST.search(question) is not None
    if not requests_definition and DEFINITION_CLAUSE.search(document):
        return -DEFINITION_CLAUSE_PENALTY

    normative_terms = {"shall", "must", "required", "prohibited", "prohibition"}
    document_terms = set(keyword_tokens(document))
    return QUESTION_TYPE_MATCH_BONUS if document_terms & normative_terms else 0.0


def _condition_adjustment(
    question: str,
    document: str,
    document_terms: set[str],
) -> float:
    adjustment = 0.0
    if document_terms & CONDITION_SIGNALS:
        adjustment += CONDITION_CLAUSE_BONUS
        if document_terms & DIRECT_RULE_TERMS:
            adjustment += DIRECT_CONDITION_OBLIGATION_BONUS
    if {"corrective", "powers"} <= document_terms:
        adjustment -= ENFORCEMENT_CLAUSE_PENALTY
    if {"codes", "conduct"} <= document_terms:
        adjustment -= GUIDANCE_CLAUSE_PENALTY

    question_terms = set(keyword_tokens(question))
    adjustment += _condition_role_adjustment(question_terms, document, document_terms)
    adjustment += _deadline_adjustment(question_terms, document, document_terms)
    return adjustment


def _condition_role_adjustment(
    question_terms: set[str],
    document: str,
    document_terms: set[str],
) -> float:
    adjustment = 0.0
    asks_about_people = bool(question_terms & PERSON_TERMS)
    asks_to_inform = bool(question_terms & INFORM_TERMS)
    if asks_about_people and asks_to_inform:
        if DATA_SUBJECT_COMMUNICATION_TERMS <= document_terms:
            adjustment += DATA_SUBJECT_COMMUNICATION_BONUS
        if SUPERVISORY_NOTIFICATION_TERMS <= document_terms:
            adjustment -= SUPERVISORY_NOTIFICATION_PENALTY

    asks_controller_to_notify_authority = {
        "controller",
        "notify",
        "supervisory",
        "authority",
    } <= question_terms
    if not asks_controller_to_notify_authority:
        return adjustment

    normalized_document = " ".join(document.casefold().split())
    if CONTROLLER_NOTIFIES_SUPERVISORY_AUTHORITY.search(normalized_document):
        return adjustment + ACTOR_ACTION_TARGET_BONUS
    if SUPERVISORY_AUTHORITY_NOTIFIES_CONTROLLER.search(normalized_document):
        return adjustment - REVERSED_ACTOR_ACTION_TARGET_PENALTY
    return adjustment


def _deadline_adjustment(
    question_terms: set[str],
    document: str,
    document_terms: set[str],
) -> float:
    if "deadline" not in question_terms:
        return 0.0

    has_explicit_timeframe = bool(document_terms & TIMEFRAME_TERMS)
    has_explicit_timeframe = (
        has_explicit_timeframe or "not later" in document.casefold()
    )
    has_explicit_timeframe = has_explicit_timeframe or "within" in document_terms
    if has_explicit_timeframe:
        return EXPLICIT_TIMEFRAME_BONUS
    return -MISSING_TIMEFRAME_PENALTY


def bm25_scores(question: str, documents: list[str]) -> np.ndarray:
    query = query_tokens(question)
    if not query or not documents:
        return np.zeros(len(documents), dtype=np.float32)
    tokenized = [keyword_tokens(document) for document in documents]
    frequencies = [Counter(tokens) for tokens in tokenized]
    lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
    average_length = float(lengths.mean()) or 1.0
    scores = np.zeros(len(documents), dtype=np.float32)
    for token in query:
        frequency = sum(token in values for values in frequencies)
        idf = math.log(
            1 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
        )
        for index, values in enumerate(frequencies):
            term_frequency = values.get(token, 0)
            if term_frequency:
                norm = 1 - BM25_B + BM25_B * (lengths[index] / average_length)
                numerator = term_frequency * (BM25_K1 + 1)
                denominator = term_frequency + BM25_K1 * norm
                scores[index] += idf * numerator / denominator
    return scores


def phrase_match_scores(question: str, documents: list[str]) -> np.ndarray:
    query = query_tokens(question)
    phrases = {
        " ".join(query[start : start + length])
        for length in range(2, min(4, len(query)) + 1)
        for start in range(len(query) - length + 1)
        if not set(query[start : start + length]) <= COMPARISON_META_TERMS
    }
    scores = np.asarray(
        [
            max(
                (
                    len(phrase.split())
                    for phrase in phrases
                    if phrase in " ".join(keyword_tokens(document))
                ),
                default=0,
            )
            for document in documents
        ],
        dtype=np.float32,
    )
    return scores / float(scores.max()) if float(scores.max()) > 0 else scores


def document_name_match(question: str, file_name: str) -> float:
    name_tokens = document_name_tokens(file_name)
    query = set(query_tokens(question))
    return len(query & name_tokens) / len(name_tokens) if name_tokens else 0.0


def document_name_tokens(file_name: str) -> set[str]:
    return {
        token
        for token in keyword_tokens(file_name)
        if token not in DOCUMENT_NAME_STOP_WORDS and not token.isdigit()
    }


def mentions_multiple_documents(question: str, file_names: list[str]) -> bool:
    return sum(document_name_match(question, name) > 0 for name in file_names) >= 2


def explicit_article_number(question: str, file_name: str) -> str | None:
    name_tokens = document_name_tokens(file_name)
    for match in ARTICLE_CITATION.finditer(question):
        prefix = question[max(0, match.start() - 50) : match.start()]
        if set(query_tokens(prefix)) & name_tokens:
            return match.group(1)
    return None


def article_anchor_index(
    article_number: str,
    chunks: list[dict],
    indices: list[int],
    ranking_scores: np.ndarray,
) -> int | None:
    article = f"article {article_number}".casefold()
    matches = [
        index
        for index in indices
        if str(chunks[index]["article"]).casefold() == article
    ]
    if not matches:
        return None
    headings = [
        index
        for index in matches
        if chunks[index]["paragraph"] is None
        and chunks[index]["point"] is None
        and chunks[index]["subpoint"] is None
    ]
    return max(headings or matches, key=lambda index: float(ranking_scores[index]))


def explicit_query_anchor(
    question: str,
    chunks: list[dict],
    ranking_scores: np.ndarray,
) -> int | None:
    document_indices: dict[int, list[int]] = {}
    for index, chunk in enumerate(chunks):
        document_indices.setdefault(chunk["document_id"], []).append(index)
    anchors = []
    for indices in document_indices.values():
        article = explicit_article_number(
            question,
            chunks[indices[0]]["original_name"],
        )
        if article is not None:
            anchor = article_anchor_index(article, chunks, indices, ranking_scores)
            if anchor is not None:
                anchors.append(anchor)
    if anchors:
        return max(anchors, key=lambda index: float(ranking_scores[index]))
    citations = ARTICLE_CITATION.findall(question)
    if len(citations) != 1:
        return None
    return article_anchor_index(
        citations[0][0],
        chunks,
        list(range(len(chunks))),
        ranking_scores,
    )


def comparison_anchor_indices(
    question: str,
    chunks: list[dict],
    searchable_documents: list[str],
    similarities: np.ndarray,
    ranking_scores: np.ndarray,
    min_similarity: float,
    maximum_documents: int,
) -> list[int]:
    by_document: dict[int, list[int]] = {}
    for index, chunk in enumerate(chunks):
        by_document.setdefault(chunk["document_id"], []).append(index)
    candidates: list[tuple[bool, float, int]] = []
    for indices in by_document.values():
        explicit_article = explicit_article_number(
            question,
            chunks[indices[0]]["original_name"],
        )
        anchor = (
            article_anchor_index(explicit_article, chunks, indices, ranking_scores)
            if explicit_article
            else None
        )
        if anchor is None:
            anchor = max(
                indices,
                key=lambda index: (
                    float(ranking_scores[index])
                    + legal_authority_adjustment(chunks[index])
                ),
            )
        elif not any(
            paragraph for _, paragraph in ARTICLE_CITATION.findall(question)
        ):
            governing = [
                index for index in indices
                if chunks[index]["article"] == chunks[anchor]["article"]
                and chunks[index]["paragraph"] == "1"
                and chunks[index]["point"] is None
                and chunks[index]["subpoint"] is None
            ]
            if governing:
                anchor = max(governing, key=lambda index: float(ranking_scores[index]))
        name_match = document_name_match(question, chunks[anchor]["original_name"])
        relevant = passes_relevance_gate(
            question,
            searchable_documents[anchor],
            float(similarities[anchor]),
            0.0,
        )
        has_weak_implicit_match = (
            explicit_article is None
            and float(ranking_scores[anchor]) < min_similarity
        )
        has_unrelated_document = name_match == 0 and not relevant
        if has_weak_implicit_match or has_unrelated_document:
            continue

        score = (
            float(ranking_scores[anchor])
            + DOCUMENT_NAME_MATCH_BONUS * name_match
        )
        candidates.append((name_match > 0, score, anchor))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    mentioned = [candidate for candidate in candidates if candidate[0]]
    is_comparison = any(pattern.search(question) for pattern in COMPARISON_PATTERNS)
    if not is_comparison and len(mentioned) < 2:
        return []
    selected = mentioned if len(mentioned) >= 2 else candidates
    if len(selected) < 2:
        return []
    return [anchor for *_, anchor in selected[:maximum_documents]]


def legal_authority_adjustment(chunk: dict) -> float:
    if chunk["article"] is None:
        section = str(chunk.get("section") or "").casefold()
        return -PREAMBLE_PENALTY if "preamble" in section else 0.0
    return OPERATIVE_CLAUSE_BONUS + normative_clause_bonus(chunk)


def normative_clause_bonus(chunk: dict) -> float:
    if chunk["article"] is None:
        return 0.0
    context = " ".join(
        part for part in (chunk.get("section"), chunk["content"]) if part
    )
    tokens = set(keyword_tokens(context))
    return NORMATIVE_CLAUSE_BONUS if tokens & NORMATIVE_TERMS else 0.0
