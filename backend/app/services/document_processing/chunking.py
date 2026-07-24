import re
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


TARGET_TOKENS = 250
MAX_TOKENS = 350
OVERLAP_TOKENS = 40
HEADING_MAX_TOKENS = 20
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
SENTENCE_END = re.compile(r"[.!?]$")


@dataclass
class TextChunk:
    content: str
    token_count: int
    page: int | None = None
    section: str | None = None
    article: str | None = None
    paragraph: str | None = None
    point: str | None = None
    subpoint: str | None = None


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def chunk_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    page: int | None = None,
    section: str | None = None,
    article: str | None = None,
    paragraph: str | None = None,
    point: str | None = None,
    subpoint: str | None = None,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    group_paragraphs: bool = False,
) -> list[TextChunk]:
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("Chunk token limits are invalid.")

    paragraphs = [
        paragraph.strip()
        for paragraph in PARAGRAPH_BREAK.split(text)
        if paragraph.strip()
    ]
    units: list[str] = []
    pending_headings: list[str] = []

    if group_paragraphs and paragraphs:
        units.append("\n\n".join(paragraphs))

    for paragraph in (paragraphs if not group_paragraphs else []):
        paragraph_token_count = len(
            tokenizer.encode(paragraph, add_special_tokens=False).ids
        )
        looks_like_heading = (
            paragraph_token_count <= HEADING_MAX_TOKENS
            and SENTENCE_END.search(paragraph) is None
        )

        if looks_like_heading:
            pending_headings.append(paragraph)
            continue

        units.append("\n\n".join([*pending_headings, paragraph]))
        pending_headings.clear()

    if pending_headings:
        units.append("\n\n".join(pending_headings))

    chunks: list[TextChunk] = []

    def append_chunk(token_ids: list[int]) -> None:
        content = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    token_count=len(token_ids),
                    page=page,
                    section=section,
                    article=article,
                    paragraph=paragraph,
                    point=point,
                    subpoint=subpoint,
                )
            )

    for unit in units:
        token_ids = tokenizer.encode(unit, add_special_tokens=False).ids
        if len(token_ids) <= max_tokens:
            append_chunk(token_ids)
            continue

        step = target_tokens - overlap_tokens
        for start in range(0, len(token_ids), step):
            append_chunk(token_ids[start : start + target_tokens])
            if start + target_tokens >= len(token_ids):
                break

    return chunks
