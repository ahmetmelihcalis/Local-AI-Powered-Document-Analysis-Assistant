from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.database import DATABASE_PATH, get_connection


@dataclass
class ChunkInput:
    content: str
    embedding: list[float] | np.ndarray
    page_number: int | None = None
    section: str | None = None


def _embedding_to_blob(embedding: list[float] | np.ndarray) -> bytes:
    vector = np.asarray(embedding, dtype=np.float32)

    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding must be a non-empty one-dimensional vector.")

    return vector.tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def create_document(
    *,
    original_name: str,
    stored_name: str,
    file_type: str,
    file_size: int,
    content_hash: str,
    status: str = "processing",
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any]:
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                original_name,
                stored_name,
                file_type,
                file_size,
                content_hash,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                original_name,
                stored_name,
                file_type,
                file_size,
                content_hash,
                status,
            ),
        )
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return dict(row)


def list_documents(database_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with get_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC, id DESC"
        ).fetchall()

    return [dict(row) for row in rows]


def get_document(
    document_id: int,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

    return dict(row) if row else None


def get_document_by_hash(
    content_hash: str,
    database_path: Path = DATABASE_PATH,
) -> dict[str, Any] | None:
    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()

    return dict(row) if row else None


def update_document_status(
    document_id: int,
    status: str,
    *,
    chunk_count: int | None = None,
    error_message: str | None = None,
    database_path: Path = DATABASE_PATH,
) -> bool:
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET status = ?,
                chunk_count = COALESCE(?, chunk_count),
                error_message = ?
            WHERE id = ?
            """,
            (status, chunk_count, error_message, document_id),
        )

    return cursor.rowcount > 0


def delete_document(
    document_id: int,
    database_path: Path = DATABASE_PATH,
) -> bool:
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )

    return cursor.rowcount > 0


def add_chunks(
    document_id: int,
    chunks: list[ChunkInput],
    database_path: Path = DATABASE_PATH,
) -> int:
    if not chunks:
        return 0

    dimensions = {
        np.asarray(chunk.embedding).size
        for chunk in chunks
    }
    if len(dimensions) != 1:
        raise ValueError("All embeddings must have the same dimensions.")

    rows = [
        (
            document_id,
            chunk_index,
            chunk.content,
            _embedding_to_blob(chunk.embedding),
            chunk.page_number,
            chunk.section,
        )
        for chunk_index, chunk in enumerate(chunks)
    ]

    with get_connection(database_path) as connection:
        connection.executemany(
            """
            INSERT INTO chunks (
                document_id,
                chunk_index,
                content,
                embedding,
                page_number,
                section
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.execute(
            "UPDATE documents SET chunk_count = ? WHERE id = ?",
            (len(chunks), document_id),
        )

    return len(chunks)


def get_chunks(
    document_id: int | None = None,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    query = """
        SELECT
            chunks.*,
            documents.original_name
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
    """
    parameters: tuple[int, ...] = ()

    if document_id is not None:
        query += " WHERE chunks.document_id = ?"
        parameters = (document_id,)

    query += " ORDER BY chunks.document_id, chunks.chunk_index"

    with get_connection(database_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    chunks = [dict(row) for row in rows]
    for chunk in chunks:
        chunk["embedding"] = _blob_to_embedding(chunk["embedding"])

    return chunks
