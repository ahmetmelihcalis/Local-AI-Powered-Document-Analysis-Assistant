from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.repositories.document_repository import list_documents
from app.services.document_ingestion import (
    DocumentIngestionError,
    DuplicateDocumentError,
    delete_ingested_document,
    ingest_document,
)


router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: int
    originalName: str
    fileType: str
    fileSize: int
    language: str | None
    status: str
    chunkCount: int
    errorMessage: str | None
    createdAt: str


def _document_response(document: dict[str, Any]) -> DocumentResponse:
    return DocumentResponse(
        id=document["id"],
        originalName=document["original_name"],
        fileType=document["file_type"],
        fileSize=document["file_size"],
        language=document["language"],
        status=document["status"],
        chunkCount=document["chunk_count"],
        errorMessage=document["error_message"],
        createdAt=document["created_at"],
    )


@router.get("", response_model=list[DocumentResponse])
def get_documents() -> list[DocumentResponse]:
    return [_document_response(document) for document in list_documents()]


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    filename = file.filename or ""
    content = await file.read()

    try:
        document = await run_in_threadpool(
            ingest_document,
            filename,
            content,
            content_type=file.content_type,
        )
    except DuplicateDocumentError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except DocumentIngestionError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    finally:
        await file.close()

    return _document_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: int) -> None:
    if not delete_ingested_document(document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
