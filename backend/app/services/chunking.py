import re
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


TARGET_TOKENS = 250
MAX_TOKENS = 350
OVERLAP_TOKENS = 40


@dataclass
class TextChunk:
    content: str
    token_count: int
    page: int | None = None
    section: str | None = None


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def chunk_text(
    text: str,
    tokenizer: Tokenizer,
    *,
    page: int | None = None,
    section: str | None = None,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[TextChunk]:
    if not 0 <= overlap_tokens < target_tokens <= max_tokens:
        raise ValueError("Chunk token limits are invalid.")

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segment_ids: list[list[int]] = []

    segment_limit = max_tokens - overlap_tokens

    for paragraph in paragraphs:
        token_ids = tokenizer.encode(paragraph, add_special_tokens=False).ids
        if len(token_ids) <= segment_limit:
            segment_ids.append(token_ids)
            continue

        segment_ids.extend(
            token_ids[start : start + segment_limit]
            for start in range(0, len(token_ids), segment_limit)
        )

    chunks: list[TextChunk] = []
    current: list[int] = []

    def append_current() -> None:
        if not current:
            return
        content = tokenizer.decode(current, skip_special_tokens=True).strip()
        if content:
            chunks.append(
                TextChunk(
                    content=content,
                    token_count=len(current),
                    page=page,
                    section=section,
                )
            )

    for segment in segment_ids:
        if current and len(current) + len(segment) > max_tokens:
            append_current()
            current = current[-overlap_tokens:] if overlap_tokens else []

        if current and len(current) >= target_tokens:
            append_current()
            current = current[-overlap_tokens:] if overlap_tokens else []

        current.extend(segment)

    append_current()
    return chunks
