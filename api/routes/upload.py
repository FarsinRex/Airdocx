import hashlib
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from datetime import datetime
from fastapi import Request
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from document_registry import register_document, get_document
from models import UploadResponse

router = APIRouter()
UPLOAD_DIR = os.environ.get("UPLOAD_DIR","/uploads")
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB hard limit
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    vs = request.app.state.vector_store
    # validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted."
        )

    # read file bytes once for size check and ID generation
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds 20MB limit. Received {file_size / (1024*1024):.1f}MB."
        )

    # deterministic document_id from filename
    document_id = hashlib.md5(file.filename.encode()).hexdigest()

    # check if already uploaded
    existing = get_document(document_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already uploaded. document_id: {document_id}"
        )

    # save to disk temporarily for processing
    temp_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        # process PDF
        processor = PDFProcessor(chunk_size=512, chunk_overlap=64)
        chunks = processor.process_pdf(temp_path)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the PDF. File may be scanned or image-based."
            )

        # embed and store
        vs = VectorStore()
        chunks = vs.embed_chunks(chunks)
        vs.upsert_chunks(chunks, namespace=document_id)

        # register in JSON
        register_document(
            document_id=document_id,
            filename=file.filename,
            chunk_count=len(chunks),
            file_size_bytes=file_size
        )

    except HTTPException:
        raise
    except Exception as e:
        # clean up temp file if anything fails
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunk_count=len(chunks),
        uploaded_at=datetime.utcnow(),
        file_size_bytes=file_size
    )