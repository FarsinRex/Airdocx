import json
import os
from typing import Dict, Optional
from datetime import datetime
from models import DocumentRecord

REGISTRY_PATH = "document_registry.json"


def _load() -> Dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"documents": {}}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _save(data: Dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


def register_document(
    document_id: str,
    filename: str,
    chunk_count: int,
    file_size_bytes: int
):
    data = _load()
    data["documents"][document_id] = {
        "document_id": document_id,
        "filename": filename,
        "namespace": document_id,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.utcnow().isoformat(),
        "file_size_bytes": file_size_bytes
    }
    _save(data)


def get_document(document_id: str) -> Optional[Dict]:
    data = _load()
    return data["documents"].get(document_id)


def list_documents() -> Dict:
    return _load()


def delete_document(document_id: str) -> bool:
    data = _load()
    if document_id not in data["documents"]:
        return False
    del data["documents"][document_id]
    _save(data)
    return True


def document_exists(document_id: str) -> bool:
    data = _load()
    return document_id in data["documents"]