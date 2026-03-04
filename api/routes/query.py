from fastapi import APIRouter, HTTPException
from models import QueryRequest, QueryResponse, ContextChunk
from rag_chain import RAGChain
from document_registry import document_exists

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):

    if not document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"No document found with id '{request.document_id}'. Upload it first."
        )

    rag = RAGChain(
        namespace=request.document_id,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )

    result = rag.answer(request.question)

    return QueryResponse(
        answer=result['answer'],
        sources=result['sources'],
        context_used=[
            ContextChunk(
                chunk_id=c['chunk_id'],
                score=round(c['score'], 4),
                pages=c['pages']
            )
            for c in result['context_used']
        ],
        model=result['model']
    )