import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.routes import upload, query, documents

load_dotenv()

app = FastAPI(
    title="PDF RAG API",
    description="Upload PDFs and query them using RAG with Groq and Pinecone.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(upload.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Query"])
app.include_router(documents.router, tags=["Documents"])


@app.get("/health")
async def health():
    return {"status": "ok"}