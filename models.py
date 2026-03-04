#for pydantic type-hint models
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# --- Upload ---

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    uploaded_at: datetime
    file_size_bytes: int


# --- Query ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: str = Field(..., description="ID returned from /upload")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)
    score_threshold: Optional[float] = Field(default=0.72, ge=0.0, le=1.0)


class ContextChunk(BaseModel):
    chunk_id: str
    score: float
    pages: List[int]


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    context_used: List[ContextChunk]
    model: str


# --- Documents ---

class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    namespace: str
    chunk_count: int
    uploaded_at: datetime
    file_size_bytes: int


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentRecord]


class DeleteResponse(BaseModel):
    message: str
    document_id: str