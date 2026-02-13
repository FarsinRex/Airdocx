# test_pipeline.py - Test complete PDF → Pinecone pipeline
import sys
from pdf_processor import PDFProcessor
from vector_store import VectorStore

def test_pipeline(pdf_path: str):
    print(" TESTING COMPLETE PIPELINE"+"\n")
    
    print("\n Step 1: Processing PDF...")
    processor = PDFProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.process_pdf(pdf_path)
    
    print("\n Step 2: Initializing Vector Store...")
    vs = VectorStore()
    
    print("\n Step 3: Embedding chunks...")
    chunks = vs.embed_chunks(chunks)
    
    print("\n Step 4: Uploading to Pinecone...")
    vs.upsert_chunks(chunks, source=pdf_path)
    
    print("\n Step 5: Testing search...")
    test_query = "What is this document about?"
    results = vs.search(test_query, top_k=3)
    
    print("\n Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.3f}")
        print(f"   {result['text'][:150]}...")
    
    # Stats
    stats = vs.get_stats()
    print(" PIPELINE TEST COMPLETE")
    print("="*60)
    print(f"Total vectors in index: {stats['total_vector_count']}")
    print(f"Chunks processed: {len(chunks)}"+"\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <pdf_file>")
        sys.exit(1)
    
    test_pipeline(sys.argv[1])