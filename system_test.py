import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
PDF_PATH = os.path.join(PROJECT_ROOT, "uploads", "upload_1.pdf")
TEST_NAMESPACE = "test_upload_1"

results = []

def passed(stage: str, detail: str = ""):
    msg = f"  [PASS] {stage}" + (f" — {detail}" if detail else "")
    print(msg)
    results.append(("PASS", stage))

def failed(stage: str, detail: str = ""):
    msg = f"  [FAIL] {stage}" + (f" — {detail}" if detail else "")
    print(msg)
    results.append(("FAIL", stage))

def header(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def summary():
    header("SUMMARY")
    total = len(results)
    passed_count = sum(1 for r in results if r[0] == "PASS")
    failed_count = total - passed_count
    for status, stage in results:
        print(f"  [{status}] {stage}")
    print(f"\n  Total: {total} | Passed: {passed_count} | Failed: {failed_count}")
    if failed_count > 0:
        print("\n  System is not ready. Fix failed stages before running the pipeline.")
    else:
        print("\n  All checks passed. Proceeding to pipeline.")
    return failed_count

#---------------------------------------#
header("STAGE 1 — Package Imports")

try:
    import pdfplumber
    passed("pdfplumber")
except ImportError as e:
    failed("pdfplumber", str(e))

try:
    from sentence_transformers import SentenceTransformer
    passed("sentence_transformers")
except ImportError as e:
    failed("sentence_transformers", str(e))

try:
    from pinecone import Pinecone
    passed("pinecone")
except ImportError as e:
    failed("pinecone", str(e))

try:
    from groq import Groq
    passed("groq")
except ImportError as e:
    failed("groq", str(e))

try:
    from fastapi import FastAPI
    passed("fastapi")
except ImportError as e:
    failed("fastapi", str(e))

try:
    from pydantic import BaseModel
    passed("pydantic")
except ImportError as e:
    failed("pydantic", str(e))

try:
    from dotenv import load_dotenv
    passed("python-dotenv")
except ImportError as e:
    failed("python-dotenv", str(e))

try:
    import uvicorn
    passed("uvicorn")
except ImportError as e:
    failed("uvicorn", str(e))

#-----------------------------------#
header("STAGE 2 — Environment Variables")

env_path = os.path.join(BACKEND_DIR, ".env")

if os.path.exists(env_path):
    passed(".env file found")
else:
    failed(".env file found", f"expected at {env_path}")

from dotenv import load_dotenv
load_dotenv(env_path)

required_keys = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GROQ_API_KEY"]
for key in required_keys:
    value = os.getenv(key)
    if value and len(value.strip()) > 0:
        passed(f"{key} is set")
    else:
        failed(f"{key} is set", "missing or empty")

#--------------------------------------------------#
header("STAGE 3 — Pinecone Connection")

pinecone_ok = False
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    stats = index.describe_index_stats()
    passed("Pinecone connection established")
    pinecone_ok = True

    dim = stats.get("dimension")
    if dim == 384:
        passed("Index dimension", f"{dim} — matches all-MiniLM-L6-v2")
    else:
        failed("Index dimension", f"expected 384, got {dim}")

except Exception as e:
    failed("Pinecone connection", str(e))
    
#-----------------------------------------------------#
header("STAGE 4 — Embedding Model")

embedder_ok = False
embedder = None
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    passed("SentenceTransformer loaded")
    embedder_ok = True
except Exception as e:
    failed("SentenceTransformer loaded", str(e))

if embedder_ok:
    try:
        test_embedding = embedder.encode("this is a test sentence").tolist()
        dim = len(test_embedding)
        if dim == 384:
            passed("Embedding dimension", f"{dim}")
        else:
            failed("Embedding dimension", f"expected 384, got {dim}")
    except Exception as e:
        failed("Embedding test", str(e))
        
#----------------------------------------------------#
header("STAGE 5 — PDF File Check")

pdf_ok = False
if os.path.exists(PDF_PATH):
    size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    passed("PDF file found", f"{size_mb:.2f}MB at {PDF_PATH}")
    if size_mb <= 20:
        passed("PDF size within 20MB limit", f"{size_mb:.2f}MB")
        pdf_ok = True
    else:
        failed("PDF size within 20MB limit", f"{size_mb:.2f}MB exceeds limit")
else:
    failed("PDF file found", f"no file at {PDF_PATH}")
    
#-----------------------------------------------------#
header("STAGE6 - PDF processor chcek")

chunks = []
processor_ok = False

if pdf_ok:
    try:
        from pdf_processor import PDFProcessor
        passed("PDFProcessor import")

        processor = PDFProcessor(chunk_size=512, chunk_overlap=64)
        chunks = processor.process_pdf(PDF_PATH)

        if len(chunks) > 0:
            passed("PDF extracted and chunked", f"{len(chunks)} chunks created")
            processor_ok = True
        else:
            failed("PDF extracted and chunked", "0 chunks returned — file may be image-based")

    except Exception as e:
        failed("PDFProcessor", str(e))

    if processor_ok:
        required_chunk_keys = ['chunk_id', 'text', 'word_count', 'source', 'pages', 'chunk_index']
        sample = chunks[0]
        missing = [k for k in required_chunk_keys if k not in sample]
        if not missing:
            passed("Chunk structure", f"all required keys present")
        else:
            failed("Chunk structure", f"missing keys: {missing}")

        avg_words = sum(c['word_count'] for c in chunks) / len(chunks)
        passed("Chunk stats", f"avg word count: {avg_words:.0f} | total chunks: {len(chunks)}")
else:
    failed("PDF Processor — skipped", "PDF file not found")
    
    

    