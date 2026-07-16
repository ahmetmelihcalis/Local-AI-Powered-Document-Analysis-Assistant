"""Parse the structural hierarchy of legal documents before token chunking."""

from dataclasses import dataclass, field

from tokenizers import Tokenizer

from app.services.chunking import TextChunk, chunk_text
from app.services.document_readers import PageText


ROMAN_NUMERAL_CHARACTERS = frozenset("IVXLCDM")


@dataclass
class LegalBlock:
    text: str
    page: int
    section: str
    article: str | None = None
    paragraph: str | None = None
    point: str | None = None
    subpoint: str | None = None


def _is_roman_numeral_or_number(value: str) -> bool:
    return value.isdigit() or (
        bool(value) and set(value.upper()) <= ROMAN_NUMERAL_CHARACTERS
    )


def _is_article_heading(line: str) -> bool:
    words = line.split()
    if len(words) != 2 or words[0].casefold() != "article":
        return False

    article_number = words[1]
    return article_number.isdigit() or (
        len(article_number) > 1
        and article_number[:-1].isdigit()
        and article_number[-1].isalpha()
    )


def _is_chapter_heading(line: str) -> bool:
    words = line.split()
    return (
        len(words) == 2
        and words[0].casefold() == "chapter"
        and _is_roman_numeral_or_number(words[1])
    )


def _is_section_heading(line: str) -> bool:
    words = line.split()
    return (
        len(words) == 2
        and words[0].casefold() == "section"
        and words[1].isdigit()
    )


def _is_annex_heading(line: str) -> bool:
    words = line.split()
    return bool(words) and words[0].casefold() == "annex" and (
        len(words) == 1
        or (len(words) == 2 and _is_roman_numeral_or_number(words[1]))
    )


def _is_page_counter(value: str) -> bool:
    current_page, separator, total_pages = value.partition("/")
    return bool(separator) and current_page.isdigit() and total_pages.isdigit()


def _is_celex_header(line: str) -> bool:
    document_code, separator, remainder = line.partition(" ")
    return (
        bool(separator)
        and len(document_code) == 10
        and document_code[:5].isdigit()
        and document_code[5].isalpha()
        and document_code[6:].isdigit()
        and remainder.startswith("—")
    )


def _is_eur_lex_page_metadata(line: str) -> bool:
    uppercase_line = line.upper()
    if uppercase_line == "EN" or uppercase_line.startswith(
        ("OJ L,", "EN OJ L,", "ELI: HTTP://", "ELI: HTTPS://")
    ):
        return True

    first_part, _, remainder = line.partition(" ")
    if _is_page_counter(first_part) and (
        not remainder or remainder.upper().startswith("ELI:")
    ):
        return True

    if _is_celex_header(line):
        return True

    return (
        line.startswith(("▼", "►"))
        and line[1:].isalnum()
        and line[1:] == line[1:].upper()
    )


def is_legal_document(pages: list[PageText]) -> bool:
    article_count = sum(
        1
        for page in pages
        for line in page.text.splitlines()
        if _is_article_heading(" ".join(line.split()))
    )
    return article_count >= 3


def _normalize_section_heading(line: str) -> str:
    words = line.split()
    if words and words[-1].isdigit() and "".join(words[:-1]).casefold() == "section":
        return f"SECTION {words[-1]}"
    return line


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    continued_word = ""

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        joins_next_line = raw_line.endswith("\u00ad")
        line = " ".join(raw_line.replace("\u00ad", "").split())

        if continued_word:
            line = f"{continued_word}{line}"
            continued_word = ""

        if joins_next_line:
            continued_word = line
            continue

        line = _normalize_section_heading(line)
        if not line or _is_eur_lex_page_metadata(line):
            continue
        lines.append(line)

    if continued_word:
        lines.append(continued_word)

    return lines


def _looks_like_article_title(line: str) -> bool:
    stripped_line = line.lstrip()
    starts_with_number = stripped_line[:1].isdigit()
    starts_with_parenthesized_number = (
        stripped_line.startswith("(") and stripped_line[1:2].isdigit()
    )
    return (
        len(line.split()) <= 20
        and not starts_with_number
        and not starts_with_parenthesized_number
    )


def _paragraph_marker(line: str) -> str | None:
    if line.startswith("("):
        marker, separator, _ = line[1:].partition(")")
        if separator and marker.isdigit():
            return marker

    marker, separator, _ = line.partition(".")
    if separator and marker.isdigit():
        return marker
    return None


def _parenthetical_marker(line: str) -> str | None:
    if not line.startswith("("):
        return None

    marker, separator, _ = line[1:].partition(")")
    if (
        not separator
        or not marker.isalpha()
        or not marker.islower()
        or len(marker) > 4
    ):
        return None
    return marker


def _is_roman_subpoint(marker: str) -> bool:
    return len(marker) > 1 and set(marker.upper()) <= ROMAN_NUMERAL_CHARACTERS


@dataclass
class _LegalStructure:
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    article_title: str | None = None
    paragraph: str | None = None
    point: str | None = None
    subpoint: str | None = None
    annex: str | None = None

    @property
    def name(self) -> str:
        article = self.article
        if article and self.article_title:
            article = f"{article} — {self.article_title}"
        parts = (self.chapter, self.section, article, self.annex)
        return " > ".join(part for part in parts if part) or "Preamble"

    def enter_chapter(self, heading: str) -> None:
        self.chapter = heading
        self.section = None
        self.article = None
        self.article_title = None
        self._reset_clauses()
        self.annex = None

    def enter_section(self, heading: str) -> None:
        self.section = heading
        self.article = None
        self.article_title = None
        self._reset_clauses()
        self.annex = None

    def enter_article(self, heading: str) -> None:
        self.article = heading
        self.article_title = None
        self._reset_clauses()
        self.annex = None

    def enter_paragraph(self, marker: str) -> None:
        self.paragraph = marker
        self.point = None
        self.subpoint = None

    def enter_point(self, marker: str) -> None:
        self.point = marker
        self.subpoint = None

    def enter_subpoint(self, marker: str) -> None:
        self.subpoint = marker

    def enter_annex(self, heading: str) -> None:
        self.chapter = None
        self.section = None
        self.article = None
        self.article_title = None
        self._reset_clauses()
        self.annex = heading

    def _reset_clauses(self) -> None:
        self.paragraph = None
        self.point = None
        self.subpoint = None


@dataclass
class _LegalBlockParser:
    structure: _LegalStructure = field(default_factory=_LegalStructure)
    blocks: list[LegalBlock] = field(default_factory=list)
    pending_headings: list[str] = field(default_factory=list)
    current_lines: list[str] = field(default_factory=list)
    current_page: int = 1
    awaiting_article_title: bool = False

    def parse(self, pages: list[PageText]) -> list[LegalBlock]:
        for page in pages:
            self._start_page(page.page)
            for line in _clean_lines(page.text):
                if not self._handle_heading(line):
                    self._append_content(line)
            self._finish_page()
        return self.blocks

    def _start_page(self, page_number: int) -> None:
        self.current_page = page_number
        self.current_lines = []

    def _finish_page(self) -> None:
        if self.pending_headings and not self.current_lines:
            self.current_lines = self.pending_headings.copy()
            self.pending_headings.clear()
        self._flush_block()

    def _flush_block(self) -> None:
        text = "\n\n".join(self.current_lines).strip()
        if text:
            self.blocks.append(
                LegalBlock(
                    text=text,
                    page=self.current_page,
                    section=self.structure.name,
                    article=self.structure.article,
                    paragraph=self.structure.paragraph,
                    point=self.structure.point,
                    subpoint=self.structure.subpoint,
                )
            )
        self.current_lines = []

    def _handle_heading(self, line: str) -> bool:
        if _is_chapter_heading(line):
            self._enter_chapter(line)
        elif _is_section_heading(line):
            self._enter_section(line)
        elif _is_article_heading(line) and self.structure.annex is None:
            self._enter_article(line)
        elif _is_annex_heading(line):
            self._enter_annex(line)
        else:
            return False
        return True

    def _enter_chapter(self, heading: str) -> None:
        self._flush_block()
        self.structure.enter_chapter(heading)
        self.pending_headings = [heading]
        self._heading_changed()

    def _enter_section(self, heading: str) -> None:
        self._flush_block()
        self.structure.enter_section(heading)
        self.pending_headings.append(heading)
        self._heading_changed()

    def _enter_article(self, heading: str) -> None:
        self._flush_block()
        self.structure.enter_article(heading)
        self.current_lines = [*self.pending_headings, heading]
        self.pending_headings.clear()
        self.awaiting_article_title = True

    def _enter_annex(self, heading: str) -> None:
        self._flush_block()
        self.structure.enter_annex(heading)
        self.current_lines = [heading]
        self.pending_headings.clear()
        self._heading_changed()

    def _heading_changed(self) -> None:
        self.awaiting_article_title = False

    def _append_content(self, line: str) -> None:
        if self.awaiting_article_title:
            self._capture_article_title(line)

        paragraph = _paragraph_marker(line)
        if paragraph is not None:
            self._flush_block()
            self.structure.enter_paragraph(paragraph)
        else:
            marker = _parenthetical_marker(line)
            if marker is not None:
                starts_subpoint = self._starts_subpoint(marker)
                self._flush_block()
                if starts_subpoint:
                    self.structure.enter_subpoint(marker)
                else:
                    self.structure.enter_point(marker)

        if self.current_lines:
            self.current_lines.append(line)
        elif self.pending_headings:
            self.pending_headings.append(line)
        else:
            self.current_lines.append(line)

    def _capture_article_title(self, line: str) -> None:
        if _looks_like_article_title(line):
            self.structure.article_title = line
        self.awaiting_article_title = False

    def _starts_subpoint(self, marker: str) -> bool:
        if _is_roman_subpoint(marker):
            return True
        if marker != "i" or self.structure.point is None:
            return False
        return self.structure.subpoint is not None or self._current_clause_opens_list()

    def _current_clause_opens_list(self) -> bool:
        return bool(self.current_lines) and self.current_lines[-1].rstrip().endswith(":")


def _create_legal_blocks(pages: list[PageText]) -> list[LegalBlock]:
    return _LegalBlockParser().parse(pages)


def chunk_legal_pdf(
    pages: list[PageText],
    tokenizer: Tokenizer,
) -> list[TextChunk]:
    return [
        chunk
        for block in _create_legal_blocks(pages)
        for chunk in chunk_text(
            block.text,
            tokenizer,
            page=block.page,
            section=block.section,
            article=block.article,
            paragraph=block.paragraph,
            point=block.point,
            subpoint=block.subpoint,
            group_paragraphs=True,
        )
    ]
