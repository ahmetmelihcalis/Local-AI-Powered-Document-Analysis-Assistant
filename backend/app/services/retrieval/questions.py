"""Classify a question's intent before retrieval selects supporting context."""

import re
from dataclasses import dataclass
from enum import StrEnum


class QuestionType(StrEnum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    CONTENT = "content"
    CONDITION = "condition"
    OBLIGATION = "obligation"
    LIST = "list"
    PROCEDURE = "procedure"
    SUMMARY = "summary"
    GENERAL = "general"


COMPARISON_PATTERNS = (
    re.compile(r"\bcompar(?:e|ed|ing|ison)\b", re.IGNORECASE),
    re.compile(r"\bdiffer(?:s|ed|ent|ently|ence|ences)?\b", re.IGNORECASE),
    re.compile(r"\b(?:versus|vs\.?)\b", re.IGNORECASE),
    re.compile(r"^\s*how\s+do(?:es)?\b.+\band\b.+\b", re.IGNORECASE),
)
CONTENT_PATTERNS = (
    re.compile(r"\bwhat information\b", re.IGNORECASE),
    re.compile(r"\bwhat details\b", re.IGNORECASE),
    re.compile(r"\b(?:contain|include|consist of)\b", re.IGNORECASE),
)
CONDITION_PATTERNS = (
    re.compile(r"\bwhen\b", re.IGNORECASE),
    re.compile(r"\b(?:within|before|after|deadline|delay)\b", re.IGNORECASE),
    re.compile(r"\b(?:under what conditions|in what circumstances)\b", re.IGNORECASE),
    re.compile(r"\bwhen (?:can|is)\b", re.IGNORECASE),
    re.compile(r"\bwhat happens if\b", re.IGNORECASE),
)
OBLIGATION_PATTERNS = (
    re.compile(r"\b(?:must|shall|should|required|prohibited)\b", re.IGNORECASE),
    re.compile(r"\b(?:obligations?|dut(?:y|ies)|prohibition)\b", re.IGNORECASE),
)
PROCEDURE_PATTERNS = (
    re.compile(r"^\s*how (?:do|can) i\b", re.IGNORECASE),
    re.compile(r"^\s*how to\b", re.IGNORECASE),
    re.compile(r"\b(?:steps?|procedure|instructions?)\b", re.IGNORECASE),
)
SUMMARY_PATTERNS = (
    re.compile(r"\b(?:summari[sz]e|overview|main points?)\b", re.IGNORECASE),
)

QUESTION_REPLACEMENTS = {
    "banned": "prohibited",
    "cant": "cannot",
    "dont": "do not",
    "notifcation": "notification",
    "pls": "please",
    "theres": "there is",
    "u": "you",
}

QUESTION_EXPANSIONS = {
    "banned": ("prohibited", "prohibition"),
    "breach": ("personal data breach",),
    "deadline": ("within", "not later", "delay"),
    "person": ("individual", "data subject"),
    "report": ("notify", "notification"),
    "reporting": ("notify", "notification"),
    "tell": ("notify", "communicate", "inform"),
    "users": ("persons", "individuals", "data subjects"),
}

DETAIL_CONTEXT_TERMS = {
    "advantages",
    "analytics",
    "architecture",
    "components",
    "configuration",
    "deployment",
    "disadvantages",
    "features",
    "implementation",
    "limitations",
    "methods",
    "requirements",
    "steps",
    "technologies",
    "workflow",
}

PROHIBITION_SUBJECT_STOP_WORDS = {
    "are",
    "banned",
    "can",
    "do",
    "is",
    "me",
    "practices",
    "prohibited",
    "systems",
    "tell",
    "the",
    "use",
    "uses",
    "what",
    "which",
}


@dataclass(frozen=True)
class RetrievalPlan:
    question_type: QuestionType
    requires_complete_list: bool = False
    prefers_section_context: bool = False
    requires_balanced_sources: bool = False
    requires_broader_context: bool = False


def normalize_question(question: str) -> str:
    normalized = " ".join(question.casefold().strip(" ?.!\t\n").split())
    normalized = re.sub(
        r"\b(?:talking|speaking)\s+to\s+(?:an?\s+)?ai\b",
        "interacting with an ai system",
        normalized,
    )
    tokens = re.findall(r"[^\W_]+|[^\w\s]", normalized, flags=re.UNICODE)
    return " ".join(QUESTION_REPLACEMENTS.get(token, token) for token in tokens)


def enrich_question(question: str) -> str:
    normalized = normalize_question(question)
    expansions = [
        term
        for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
        for term in QUESTION_EXPANSIONS.get(token, ())
    ]
    return " ".join((normalized, *dict.fromkeys(expansions)))


def classify_question_type(question: str) -> QuestionType:
    normalized = normalize_question(question)
    if any(pattern.search(normalized) for pattern in COMPARISON_PATTERNS):
        return QuestionType.COMPARISON
    if any(pattern.search(normalized) for pattern in SUMMARY_PATTERNS):
        return QuestionType.SUMMARY
    if any(pattern.search(normalized) for pattern in PROCEDURE_PATTERNS):
        return QuestionType.PROCEDURE
    if re.match(r"^what is .+ about$", normalized):
        return QuestionType.SUMMARY
    if normalized.startswith("what is "):
        return QuestionType.DEFINITION
    if any(pattern.search(normalized) for pattern in CONTENT_PATTERNS):
        return QuestionType.CONTENT
    if any(pattern.search(normalized) for pattern in CONDITION_PATTERNS):
        return QuestionType.CONDITION
    if "list" in normalized:
        return QuestionType.LIST
    if any(pattern.search(normalized) for pattern in OBLIGATION_PATTERNS):
        return QuestionType.OBLIGATION
    if normalized.startswith(("what are ", "which ")):
        return QuestionType.LIST
    return QuestionType.GENERAL


def requires_broader_context(question: str) -> bool:
    normalized = normalize_question(question)
    tokens = set(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))
    detail_terms = tokens & DETAIL_CONTEXT_TERMS
    has_multiple_aspects = " and " in normalized and len(tokens) >= 8
    return len(detail_terms) >= 2 or (bool(detail_terms) and has_multiple_aspects)


def build_retrieval_plan(question: str) -> RetrievalPlan:
    question_type = classify_question_type(question)
    broader_context = requires_broader_context(question)
    if question_type == QuestionType.COMPARISON:
        return RetrievalPlan(
            question_type,
            requires_balanced_sources=True,
        )
    if question_type in {QuestionType.LIST, QuestionType.CONTENT}:
        return RetrievalPlan(question_type, requires_complete_list=True)
    normalized = normalize_question(question)
    if ("banned" in normalized or "prohibited" in normalized) and re.search(
        r"\b(?:practices|systems|activities|uses)\b", normalized
    ):
        subject_terms = {
            token
            for token in re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)
            if token not in PROHIBITION_SUBJECT_STOP_WORDS
        }
        if subject_terms <= {"ai", "system"}:
            return RetrievalPlan(question_type, requires_complete_list=True)
    if question_type in {QuestionType.PROCEDURE, QuestionType.SUMMARY}:
        return RetrievalPlan(
            question_type,
            prefers_section_context=True,
            requires_broader_context=broader_context,
        )
    return RetrievalPlan(question_type, requires_broader_context=broader_context)
