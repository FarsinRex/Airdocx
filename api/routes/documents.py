from fastapi import APIRouter, HTTPException
from models import DocumentListResponse, DocumentRecord, DeleteResponse
from document_registry import list_documents, get_document, delete_document
from vector_store import VectorStore
from datetime import datetime

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def get_all_documents():
    data = list_documents()
    docs = [
        DocumentRecord(**record)
        for record in data["documents"].values()
    ]
    return DocumentListResponse(total=len(docs), documents=docs)


@router.get("/documents/{document_id}", response_model=DocumentRecord)
async def get_single_document(document_id: str):
    record = get_document(document_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found."
        )
    return DocumentRecord(**record)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_single_document(document_id: str):
    record = get_document(document_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found."
        )

    # delete from Pinecone first
    try:
        vs = VectorStore()
        vs.delete_namespace(document_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete vectors from Pinecone: {str(e)}"
        )

    # remove from registry
    delete_document(document_id)

    return DeleteResponse(
        message=f"Document '{record['filename']}' deleted successfully.",
        document_id=document_id
    )