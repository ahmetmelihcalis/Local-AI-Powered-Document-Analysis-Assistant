from pathlib import Path
from typing import Any

from app.database import DATABASE_PATH, get_connection


def create_document(
    *,
    original_name: str,
    stored_name: str,
    file_type: str,
    file_size: int,
    content_hash: str,
    language: str | None = None,
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
                language,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_name,
                stored_name,
                file_type,
                file_size,
                content_hash,
                language,
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
