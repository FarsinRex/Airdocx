import sys
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_chain import RAGChain

CHUNK_SIZE = 64
CHUNK_OVERLAP = 10 
TEST_NAMESPACE = "test"

def test_pipeline(pdf_path: str):
    print("--- Step 1: Processing PDF ---")
    processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.process_pdf(pdf_path)
    print(f"Chunks created: {len(chunks)}")
    print(f"Sample chunk: {chunks[0]['text'][:120]}")

    print("\n--- Step 2: Initializing Vector Store ---")
    vs = VectorStore()

    print("\n--- Step 3: Embedding chunks ---")
    chunks = vs.embed_chunks(chunks)

    print("\n--- Step 4: Uploading to Pinecone ---")
    vs.upsert_chunks(chunks, namespace=TEST_NAMESPACE)

    print("\n--- Step 5: Testing retrieval ---")
    test_query = "What is the main argument of the document?"
    results = vs.search(test_query, top_k=3, namespace=TEST_NAMESPACE)

    if not results:
        print("No results above threshold. Check your document content or lower SCORE_THRESHOLD.")
    else:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f} | Pages: {result['pages']}")
            print(f"   {result['text'][:150]}...")

    print("\n--- Step 6: Testing full RAG answer ---")
    rag = RAGChain(namespace=TEST_NAMESPACE)
    output = rag.answer(test_query)
    print(f"\nAnswer: {output['answer']}")
    print(f"Sources: {output['sources']}")

    print("\n--- Step 7: Cleanup ---")
    vs.delete_namespace(TEST_NAMESPACE)
    print("Test namespace cleared.")

    stats = vs.get_stats()
    print(f"\nFinal index stats: {stats}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <pdf_file>")
        sys.exit(1)

    test_pipeline(sys.argv[1])