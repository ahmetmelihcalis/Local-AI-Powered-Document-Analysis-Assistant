import re
from pathlib import Path

from app.services.retrieval import QuestionScope, RetrievedChunk
from app.services.retrieval.questions import QuestionType, classify_question_type


INSUFFICIENT_CONTEXT_TOKEN = "INSUFFICIENT_CONTEXT"
MIN_PARENT_CLAUSE_COVERAGE = 0.30
SUFFICIENT_CONTEXT_RESPONSES = {"sufficient_context", "sufficient context"}

WORDS = re.compile(r"[^\W_]+", re.UNICODE)
ARTICLE_REFERENCE = re.compile(r"\bArticle\s+\d+[a-z]?(?:\s*\([a-z0-9ivxlcdm]+\))*", re.IGNORECASE)
SUSPICIOUS_REFERENCE = re.compile(
    r"\bs\d+[a-z]?\b|\b\d+[a-z]\([a-z0-9]+\)",
    re.IGNORECASE,
)
CHILD_LIST_MARKER = re.compile(r"^(?:\([a-z0-9ivxlcdm]+\)|\d+[.)])\s+", re.IGNORECASE)
DANGLING_LIST_MARKER = re.compile(r"(?:^|\s)(?:\d+[.)]|\([a-z0-9]+\))\s*$", re.IGNORECASE)
PARENT_LIST_INTRODUCTIONS = (
    "at least",
    "any of the following",
    "following conditions",
    "following grounds",
)

QUALIFIERS = (
    "always",
    "does not cover",
    "except",
    "in so far as",
    "shall not apply",
    "unless",
    "where applicable",
)
INSUFFICIENT_RESPONSE_PHRASES = (
    INSUFFICIENT_CONTEXT_TOKEN.casefold(),
    "do not contain enough information",
    "does not contain enough information",
    "not enough information",
    "insufficient information",
    "cannot directly access",
    "can not directly access",
    "unable to access",
    "provided document excerpts do not",
    "not stated in the provided",
    "not specified in the provided",
    "not directly supplied",
    "cannot be determined from",
    "cannot answer",
)
NORMATIVE_ACTION_FORMS = {
    "assess": {"assess", "assessed", "assesses", "assessing"},
    "communicate": {"communicate", "communicated", "communicates", "communicating"},
    "contain": {"contain", "contained", "contains", "containing"},
    "document": {"document", "documented", "documents", "documenting"},
    "ensure": {"ensure", "ensured", "ensures", "ensuring"},
    "inform": {"inform", "informed", "informs", "informing"},
    "maintain": {"maintain", "maintained", "maintains", "maintaining"},
    "notify": {"notify", "notified", "notifies", "notifying"},
    "process": {"process", "processed", "processes", "processing"},
    "prohibit": {"prohibit", "prohibited", "prohibits", "prohibiting"},
    "provide": {"provide", "provided", "provides", "providing"},
    "record": {"record", "recorded", "records", "recording"},
    "report": {"report", "reported", "reports", "reporting"},
    "retain": {"retain", "retained", "retains", "retaining"},
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "may",
    "must",
    "of",
    "on",
    "or",
    "shall",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "when",
    "which",
    "who",
    "with",
}


def legal_reference(source: RetrievedChunk) -> str | None:
    if source.article is None:
        return None

    article = source.article.strip()
    reference = (
        article
        if article.casefold().startswith("article ")
        else f"Article {article}"
    )
    for value in (source.paragraph, source.point, source.subpoint):
        if value is not None:
            reference += f"({value})"
    return reference


def is_insufficient_response(answer: str) -> bool:
    lowered = " ".join(answer.casefold().split())
    if lowered in SUFFICIENT_CONTEXT_RESPONSES:
        return True
    return any(phrase in lowered for phrase in INSUFFICIENT_RESPONSE_PHRASES)


def terms(text: str) -> list[str]:
    return [
        term.casefold()
        for term in WORDS.findall(text)
        if len(term) > 2 and term.casefold() not in STOP_WORDS
    ]


def sources_support_question(question: str, sources: list[RetrievedChunk]) -> bool:
    question_terms = set(terms(question))
    if not question_terms:
        return False

    source_text = " ".join(source.content for source in sources)
    source_terms = set(terms(source_text))
    overlap = len(question_terms & source_terms)
    required_overlap = max(1, round(len(question_terms) * 0.25))
    return overlap >= required_overlap


def validation_issues(
    answer: str,
    question: str,
    scope: QuestionScope,
    sources: list[RetrievedChunk],
) -> tuple[str, ...]:
    issues = _generic_issues(answer, question, scope, sources)

    if any(source.article is not None for source in sources):
        issues.extend(_legal_issues(answer, question, scope, sources))
    if scope == QuestionScope.COMPLETE_LIST:
        issues.extend(_list_issues(answer, sources))
    if scope == QuestionScope.CROSS_DOCUMENT:
        issues.extend(_comparison_issues(answer, sources))

    return tuple(issues)


def expected_labels(sources: list[RetrievedChunk]) -> tuple[str, ...]:
    points: list[str] = []
    point_paragraphs: set[str | None] = set()

    for source in sources:
        if source.point is not None and source.subpoint is None:
            points.append(source.point)
            point_paragraphs.add(source.paragraph)

    if points and len(point_paragraphs) == 1:
        return tuple(dict.fromkeys(f"({point})" for point in points))
    if points:
        return ()

    paragraphs = [
        source.paragraph
        for source in sources
        if source.paragraph is not None and source.point is None
    ]
    return tuple(dict.fromkeys(f"({paragraph})" for paragraph in paragraphs))


def _generic_issues(
    answer: str,
    question: str,
    scope: QuestionScope,
    sources: list[RetrievedChunk],
) -> list[str]:
    lowered = answer.casefold()
    checks = (
        ("empty answer", not answer),
        ("prompt leak", _has_prompt_leak(lowered)),
        ("repeated text", _has_repetition(answer)),
        ("dangling list marker", DANGLING_LIST_MARKER.search(answer) is not None),
        (
            "parent clause has no following child clauses",
            _has_unresolved_parent_clause(answer),
        ),
        ("unsupported legal reference", _unsupported_reference(answer, sources)),
        (
            "unrequested legal reference",
            _has_unrequested_article_reference(answer, question, scope),
        ),
    )
    return [issue for issue, detected in checks if detected]


def _legal_issues(
    answer: str,
    question: str,
    scope: QuestionScope,
    sources: list[RetrievedChunk],
) -> list[str]:
    issues = []
    source_text = " ".join(source.content for source in sources)
    source_terms = set(terms(source_text))
    answer_terms = terms(answer)
    grounded_terms = sum(term in source_terms for term in answer_terms)
    grounding = grounded_terms / max(len(answer_terms), 1)
    minimum_grounding = 0.92 if scope == QuestionScope.DEFINITION else 0.72

    if grounding < minimum_grounding:
        issues.append("insufficient grounding")
    if scope != QuestionScope.CROSS_DOCUMENT:
        issues.extend(_missing_qualifier_issues(answer, source_text))
    if classify_question_type(question) in {QuestionType.CONDITION, QuestionType.OBLIGATION}:
        issues.extend(_missing_action_issues(answer, source_text))
    if _selected_clause_is_incomplete(answer_terms, scope, sources):
        issues.append("selected clause is incomplete")
    return issues


def _list_issues(answer: str, sources: list[RetrievedChunk]) -> list[str]:
    issues = []
    parent_terms = _parent_clause_terms(sources)
    if parent_terms:
        child_terms = _child_clause_terms(sources)
        required_terms = (parent_terms - child_terms) or parent_terms
        coverage = len(set(terms(answer)) & required_terms) / len(required_terms)
        if coverage < MIN_PARENT_CLAUSE_COVERAGE:
            issues.append("missing parent clause rule or conditions")

    missing_labels = [
        label
        for label in expected_labels(sources)
        if not _has_label(answer, label)
    ]
    if missing_labels:
        issues.append("missing clauses: " + ", ".join(missing_labels))
    return issues


def _comparison_issues(answer: str, sources: list[RetrievedChunk]) -> list[str]:
    answer_terms = set(terms(answer))
    missing_documents = []
    for file_name in dict.fromkeys(source.file_name for source in sources):
        label_terms = set(terms(Path(file_name).stem))
        if label_terms and not answer_terms.intersection(label_terms):
            missing_documents.append(file_name)
    if not missing_documents:
        return []
    return ["missing compared documents: " + ", ".join(missing_documents)]


def _has_prompt_leak(lowered_answer: str) -> bool:
    return (
        "using only the supplied" in lowered_answer
        or "required labels:" in lowered_answer
        or lowered_answer.startswith("context:")
        or "\nquestion:" in lowered_answer
    )


def _has_repetition(answer: str) -> bool:
    answer_terms = terms(answer)
    ngrams = [
        tuple(answer_terms[index : index + 5])
        for index in range(len(answer_terms) - 4)
    ]
    return len(ngrams) != len(set(ngrams))


def _has_unresolved_parent_clause(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if not _is_parent_list_introduction(line):
            continue
        if not any(CHILD_LIST_MARKER.match(candidate) for candidate in lines[index + 1 :]):
            return True
    return False


def _is_parent_list_introduction(line: str) -> bool:
    normalized = line.casefold().rstrip()
    return normalized.endswith(":") and any(
        phrase in normalized for phrase in PARENT_LIST_INTRODUCTIONS
    )


def _unsupported_reference(answer: str, sources: list[RetrievedChunk]) -> bool:
    source_text = " ".join(source.content for source in sources).casefold()
    allowed_references = {
        _normalized_reference(reference)
        for source in sources
        if (reference := legal_reference(source)) is not None
    }
    for reference in ARTICLE_REFERENCE.findall(answer):
        normalized_reference = _normalized_reference(reference)
        if normalized_reference not in allowed_references and reference.casefold() not in source_text:
            return True
    return any(
        match.group(0).casefold() not in source_text
        for match in SUSPICIOUS_REFERENCE.finditer(answer)
    )


def _normalized_reference(reference: str) -> str:
    return "".join(reference.casefold().split())


def _has_unrequested_article_reference(
    answer: str,
    question: str,
    scope: QuestionScope,
) -> bool:
    answer_reference = ARTICLE_REFERENCE.search(answer)
    return (
        scope == QuestionScope.FOCUSED
        and answer_reference is not None
        and ARTICLE_REFERENCE.search(question) is None
    )


def _missing_qualifier_issues(answer: str, source_text: str) -> list[str]:
    answer_text = answer.casefold()
    source_text = source_text.casefold()
    missing = [
        qualifier
        for qualifier in QUALIFIERS
        if qualifier in source_text and qualifier not in answer_text
    ]
    if not missing:
        return []
    return ["missing qualifiers: " + ", ".join(missing)]


def _missing_action_issues(answer: str, source_text: str) -> list[str]:
    missing_actions = _normative_actions(source_text) - _normative_actions(answer)
    if not missing_actions:
        return []
    return ["missing legal action: " + ", ".join(sorted(missing_actions))]


def _selected_clause_is_incomplete(
    answer_terms: list[str],
    scope: QuestionScope,
    sources: list[RetrievedChunk],
) -> bool:
    if scope != QuestionScope.FOCUSED or len(sources) != 1:
        return False
    selected_terms = set(terms(sources[0].content))
    coverage = len(set(answer_terms) & selected_terms) / max(len(selected_terms), 1)
    return coverage < 0.45


def _parent_clause_terms(sources: list[RetrievedChunk]) -> set[str]:
    parent_text = " ".join(
        source.content
        for source in sources
        if source.paragraph is not None
        and source.point is None
        and source.subpoint is None
    )
    return set(terms(parent_text))


def _child_clause_terms(sources: list[RetrievedChunk]) -> set[str]:
    child_text = " ".join(
        source.content
        for source in sources
        if source.point is not None or source.subpoint is not None
    )
    return set(terms(child_text))


def _normative_actions(text: str) -> set[str]:
    text_terms = set(terms(text))
    return {
        action
        for action, forms in NORMATIVE_ACTION_FORMS.items()
        if text_terms & forms
    }


def _has_label(answer: str, label: str) -> bool:
    value = re.escape(label.strip("()"))
    if value.isdigit():
        marker = rf"(?:\({value}\)|{value}[.)])"
    else:
        marker = rf"\({value}\)"
    return bool(re.search(rf"(?:^|\n)\s*{marker}\s+", answer, flags=re.IGNORECASE))
