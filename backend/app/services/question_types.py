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


@dataclass(frozen=True)
class RetrievalPlan:
    question_type: QuestionType
    requires_complete_list: bool = False
    prefers_section_context: bool = False
    requires_balanced_sources: bool = False


def classify_question_type(question: str) -> QuestionType:
    normalized = " ".join(question.casefold().strip(" ?.!").split())
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


def build_retrieval_plan(question: str) -> RetrievalPlan:
    question_type = classify_question_type(question)
    if question_type == QuestionType.COMPARISON:
        return RetrievalPlan(question_type, requires_balanced_sources=True)
    if question_type == QuestionType.LIST:
        return RetrievalPlan(question_type, requires_complete_list=True)
    if question_type in {QuestionType.PROCEDURE, QuestionType.SUMMARY}:
        return RetrievalPlan(question_type, prefers_section_context=True)
    return RetrievalPlan(question_type)
