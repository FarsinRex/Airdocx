import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from .utils import save_uploaded_file, extract_text_from_pdf
from .ingest import chunk_text
from .embed_store import EmbedStore
from .query import answer_question

app = FastAPI()
store = EmbedStore()

MAX_FILE_SIZE_MB = 5
MAX_CHUNK_SIZE = 1000
MAX_CHUNKS_PER_FILE = 500
BATCH_SIZE = 50

class QueryIn(BaseModel):
    question: str
    k: int = 4

@app.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    tmp_dir = "D:/projects/airdocx/temp_uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, file.filename)
    save_uploaded_file(file, tmp_path)

    # File size check
    file_size = os.path.getsize(tmp_path) / (1024*1024)
    if file_size > MAX_FILE_SIZE_MB:
        os.remove(tmp_path)
        raise HTTPException(status_code=400,
            detail=f"File too large ({file_size:.2f} MB). Limit is {MAX_FILE_SIZE_MB} MB.")

    # Extract text
    try:
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(tmp_path)
        else:
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    # Chunk text
    def safe_chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            chunk = text[start:end]
            chunks.append((chunk, start, end))
            start = end
        return chunks

    chunks = safe_chunk_text(text)
    if len(chunks) > MAX_CHUNKS_PER_FILE:
        raise HTTPException(status_code=400,
            detail=f"Too many chunks ({len(chunks)}). Please upload a smaller file.")

    payload = [{"text": chunk, "meta": {"source": file.filename, "start": start, "end": end}}
               for chunk, start, end in chunks]

    for i in range(0, len(payload), BATCH_SIZE):
        batch = payload[i:i+BATCH_SIZE]
        store.add_texts(batch)

    return {"status": "ok", "chunks": len(payload)}

@app.post("/query/")
def query(q: QueryIn):
    try:
        res = answer_question(q.question, k=q.k)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "server is running, welcome to airdocx"}
