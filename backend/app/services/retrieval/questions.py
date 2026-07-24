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


WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
AI_INTERACTION_PATTERN = re.compile(
    r"\b(?:talking|speaking)\s+to\s+(?:an?\s+)?ai\b",
    re.IGNORECASE,
)
PROJECT_SUMMARY_PATTERN = re.compile(r"^what is .+ about$", re.IGNORECASE)
GENERAL_DOCUMENT_PATTERN = re.compile(
    r"\b(?:this|the)\s+(?:project|application|app|dashboard)\b"
    r"|\bproject\s+(?:prerequisites|requirements|setup)\b",
    re.IGNORECASE,
)
BROAD_CONDITION_PATTERN = re.compile(
    r"\bwhen\s+(?:may|can|does|do)\b|\b(?:can|may)\s+\w+|"
    r"^\s*(?:an?\s+)?ai(?:\s+system)?\s+(?:make|making)\s+decisions?\b",
    re.IGNORECASE,
)
BROAD_PROHIBITION_PATTERN = re.compile(
    r"\b(?:practices|systems|activities|uses)\b",
    re.IGNORECASE,
)

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
    re.compile(r"^\s*(?:can|may)\b", re.IGNORECASE),
    re.compile(
        r"^\s*(?:an?\s+)?ai(?:\s+system)?\s+(?:make|making)\s+decisions?\b",
        re.IGNORECASE,
    ),
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
    requires_governing_context: bool = False
    prefers_general_documents: bool = False


def normalize_question(question: str) -> str:
    normalized = " ".join(question.casefold().strip(" ?.!\t\n").split())
    normalized = AI_INTERACTION_PATTERN.sub(
        "interacting with an ai system",
        normalized,
    )
    tokens = re.findall(r"[^\W_]+|[^\w\s]", normalized, flags=re.UNICODE)
    return " ".join(QUESTION_REPLACEMENTS.get(token, token) for token in tokens)


def enrich_question(question: str) -> str:
    normalized = normalize_question(question)
    expansions = [
        term
        for token in _words(normalized)
        for term in QUESTION_EXPANSIONS.get(token, ())
    ]
    return " ".join((normalized, *dict.fromkeys(expansions)))


def classify_question_type(question: str) -> QuestionType:
    normalized = normalize_question(question)
    if _matches_any(normalized, COMPARISON_PATTERNS):
        return QuestionType.COMPARISON
    if _matches_any(normalized, SUMMARY_PATTERNS) or PROJECT_SUMMARY_PATTERN.match(normalized):
        return QuestionType.SUMMARY
    if _matches_any(normalized, PROCEDURE_PATTERNS):
        return QuestionType.PROCEDURE
    if normalized.startswith("what is "):
        return QuestionType.DEFINITION
    if _matches_any(normalized, CONTENT_PATTERNS):
        return QuestionType.CONTENT
    if _matches_any(normalized, CONDITION_PATTERNS):
        return QuestionType.CONDITION
    if "list" in normalized:
        return QuestionType.LIST
    if _matches_any(normalized, OBLIGATION_PATTERNS):
        return QuestionType.OBLIGATION
    if normalized.startswith(("what are ", "which ")):
        return QuestionType.LIST
    return QuestionType.GENERAL


def requires_broader_context(question: str) -> bool:
    normalized = normalize_question(question)
    question_terms = set(_words(normalized))
    detail_terms = question_terms & DETAIL_CONTEXT_TERMS
    has_multiple_aspects = " and " in normalized and len(question_terms) >= 8
    return len(detail_terms) >= 2 or (bool(detail_terms) and has_multiple_aspects)


def build_retrieval_plan(question: str) -> RetrievalPlan:
    question_type = classify_question_type(question)
    normalized = normalize_question(question)
    prefers_general_documents = GENERAL_DOCUMENT_PATTERN.search(normalized) is not None
    if question_type == QuestionType.COMPARISON:
        return RetrievalPlan(
            question_type,
            requires_balanced_sources=True,
            prefers_general_documents=prefers_general_documents,
        )

    requires_complete_list = (
        question_type in {QuestionType.LIST, QuestionType.CONTENT}
        or _requires_complete_prohibition_list(question)
    )
    if requires_complete_list:
        return RetrievalPlan(
            question_type,
            requires_complete_list=True,
            prefers_general_documents=prefers_general_documents,
        )

    broader_context = requires_broader_context(question)
    governing_context = (
        question_type == QuestionType.CONDITION
        and BROAD_CONDITION_PATTERN.search(normalized) is not None
    )
    if question_type in {QuestionType.PROCEDURE, QuestionType.SUMMARY}:
        return RetrievalPlan(
            question_type,
            prefers_section_context=True,
            requires_broader_context=broader_context,
            prefers_general_documents=prefers_general_documents,
        )
    return RetrievalPlan(
        question_type,
        requires_broader_context=broader_context,
        requires_governing_context=governing_context,
        prefers_general_documents=prefers_general_documents,
    )


def _requires_complete_prohibition_list(question: str) -> bool:
    normalized = normalize_question(question)
    asks_about_prohibitions = "banned" in normalized or "prohibited" in normalized
    if not asks_about_prohibitions or not BROAD_PROHIBITION_PATTERN.search(normalized):
        return False

    subject_terms = {
        token
        for token in _words(normalized)
        if token not in PROHIBITION_SUBJECT_STOP_WORDS
    }
    return subject_terms <= {"ai", "system"}


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)
