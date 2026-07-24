import re

from app.services.retrieval import RetrievedChunk


SOURCE_MARKER = re.compile(r"\[SOURCE\s+\d+(?:\s*\|[^\]]+)?\]", re.IGNORECASE)
MARKDOWN_CLEANUP_RULES = (
    (re.compile(r"```(?:\w+)?\s*"), ""),
    (re.compile(r"(?<!\w)#{1,6}\s*"), ""),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
)
TRAILING_CLAUSE_SEPARATOR = re.compile(r";\s*$")
PARENT_PARAGRAPH_MARKER = re.compile(r"^\s*\d+[.)]\s+")
CLAUSE_MARKER = re.compile(
    r"^\s*(?:\d+[.)]|\([a-z0-9ivxlcdm]+\))\s+",
    re.IGNORECASE,
)
PROCEDURE_STRUCTURE = re.compile(
    r"\b(?:quick start|prerequisites?|installation|setup|steps?)\b"
    r"|(?:^|\s)\d+[.)]\s+",
    re.IGNORECASE,
)


def clean_answer(answer: str, preserve_lines: bool) -> str:
    answer = SOURCE_MARKER.sub("", answer)
    for pattern, replacement in MARKDOWN_CLEANUP_RULES:
        answer = pattern.sub(replacement, answer)

    lines = [" ".join(line.split()) for line in answer.splitlines() if line.strip()]
    separator = "\n" if preserve_lines else " "
    answer = separator.join(lines).strip(" |")
    answer = TRAILING_CLAUSE_SEPARATOR.sub(".", answer)
    return _capitalize_first_letter(answer)


def remove_parent_paragraph_marker(answer: str) -> str:
    lines = answer.splitlines()
    if lines:
        lines[0] = PARENT_PARAGRAPH_MARKER.sub("", lines[0], count=1)
    return "\n".join(lines)


def remove_clause_marker(content: str) -> str:
    content = CLAUSE_MARKER.sub("", content.strip(), count=1)
    return _capitalize_first_letter(" ".join(content.split()))


def has_procedure_structure(sources: list[RetrievedChunk]) -> bool:
    context = "\n".join(source.content for source in sources)
    return PROCEDURE_STRUCTURE.search(context) is not None


def _capitalize_first_letter(text: str) -> str:
    if text[:1].isalpha():
        return text[:1].upper() + text[1:]
    return text
