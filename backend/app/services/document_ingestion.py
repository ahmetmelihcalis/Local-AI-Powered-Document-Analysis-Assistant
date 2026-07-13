import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from app.database import DATABASE_PATH, DATA_DIR
from app.repositories.document_repository import (
    ChunkInput,
    add_chunks,
    create_document,
    delete_document,
    get_document,
    get_document_by_hash,
    list_documents,
    update_document_status,
)
from app.services.chunking import TextChunk, chunk_text, load_tokenizer
from app.services.document_readers import (
    DocumentText,
    read_docx_document,
    read_pdf_document,
    read_text_document,
)
from app.services.foundry_service import (
    create_embeddings,
    get_embedding_tokenizer_path,
)


MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_DOCUMENTS = 20
DOCUMENTS_DIR = DATA_DIR / "documents"

ALLOWED_CONTENT_TYPES = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
}


class DocumentIngestionError(ValueError):
    pass


class DuplicateDocumentError(DocumentIngestionError):
    pass


def _validate_document(
    filename: str,
    content: bytes,
    content_type: str | None,
) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_CONTENT_TYPES:
        raise DocumentIngestionError("Only TXT, Markdown, PDF and DOCX files are supported.")

    if not content:
        raise DocumentIngestionError("The document is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise DocumentIngestionError("The document cannot be larger than 10 MB.")

    if (
        content_type
        and content_type != "application/octet-stream"
        and content_type not in ALLOWED_CONTENT_TYPES[extension]
    ):
        raise DocumentIngestionError("The file extension and content type do not match.")

    return extension


def _group_docx_blocks(blocks: list[DocumentText]) -> list[DocumentText]:
    groups: list[DocumentText] = []

    for block in blocks:
        if groups and groups[-1].section == block.section:
            groups[-1].text += f"\n\n{block.text}"
        else:
            groups.append(DocumentText(text=block.text, section=block.section))

    return groups


def _create_chunks(filename: str, content: bytes, tokenizer) -> list[TextChunk]:
    extension = Path(filename).suffix.lower()

    if extension in {".txt", ".md"}:
        return chunk_text(read_text_document(filename, content), tokenizer)

    if extension == ".pdf":
        return [
            chunk
            for page in read_pdf_document(content)
            for chunk in chunk_text(page.text, tokenizer, page=page.page)
        ]

    if extension == ".docx":
        return [
            chunk
            for block in _group_docx_blocks(read_docx_document(content))
            for chunk in chunk_text(block.text, tokenizer, section=block.section)
        ]

    raise DocumentIngestionError("The document type is not supported.")


def _create_document_embeddings(
    texts: list[str],
    embedding_function: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    embeddings = embedding_function(texts)

    if len(embeddings) != len(texts):
        raise RuntimeError("The embedding model returned an unexpected result count.")

    return embeddings


def ingest_document(
    filename: str,
    content: bytes,
    *,
    content_type: str | None = None,
    database_path: Path = DATABASE_PATH,
    documents_dir: Path = DOCUMENTS_DIR,
    embedding_function: Callable[[list[str]], list[list[float]]] = create_embeddings,
    tokenizer_path_provider: Callable[[], Path] = get_embedding_tokenizer_path,
) -> dict:
    extension = _validate_document(filename, content, content_type)
    content_hash = hashlib.sha256(content).hexdigest()

    if get_document_by_hash(content_hash, database_path):
        raise DuplicateDocumentError("This document has already been uploaded.")

    if len(list_documents(database_path)) >= MAX_DOCUMENTS:
        raise DocumentIngestionError("A maximum of 20 documents can be stored.")

    stored_name = f"{uuid4()}{extension}"
    stored_path = documents_dir / stored_name
    documents_dir.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)

    try:
        document = create_document(
            original_name=Path(filename).name,
            stored_name=stored_name,
            file_type=extension.removeprefix("."),
            file_size=len(content),
            content_hash=content_hash,
            status="processing",
            database_path=database_path,
        )
    except sqlite3.IntegrityError as error:
        stored_path.unlink(missing_ok=True)
        raise DuplicateDocumentError("This document has already been uploaded.") from error
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise

    try:
        tokenizer = load_tokenizer(tokenizer_path_provider())
        chunks = _create_chunks(filename, content, tokenizer)

        if not chunks:
            raise DocumentIngestionError("The document did not produce any readable chunks.")

        embeddings = _create_document_embeddings(
            [chunk.content for chunk in chunks],
            embedding_function,
        )
        add_chunks(
            document["id"],
            [
                ChunkInput(
                    content=chunk.content,
                    embedding=embedding,
                    page_number=chunk.page,
                    section=chunk.section,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
            database_path,
        )
        update_document_status(
            document["id"],
            "ready",
            chunk_count=len(chunks),
            database_path=database_path,
        )
        return get_document_by_hash(content_hash, database_path)
    except Exception as error:
        update_document_status(
            document["id"],
            "error",
            error_message=str(error),
            database_path=database_path,
        )
        if isinstance(error, DocumentIngestionError):
            raise
        raise DocumentIngestionError(str(error)) from error


def delete_ingested_document(
    document_id: int,
    *,
    database_path: Path = DATABASE_PATH,
    documents_dir: Path = DOCUMENTS_DIR,
) -> bool:
    document = get_document(document_id, database_path)

    if document is None:
        return False

    if not delete_document(document_id, database_path):
        return False

    stored_path = documents_dir / Path(document["stored_name"]).name
    stored_path.unlink(missing_ok=True)
    return True
