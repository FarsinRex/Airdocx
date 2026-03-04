import os
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
SCORE_THRESHOLD = 0.72

class VectorStore:
    """Manages embeddings and pinecone storage
    """
    def __init__(self):
        #initliaze embedding model 
        api_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME')
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found, check your env file")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME not found, check your env file")
        
        print("Loading embedding model")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        print("connecting to pinecone")
        
        
        #initiaze pinecone connection
        
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(index_name)
        stats = self.index.describe_index_stats()
        print(f"connected to index:{index_name} with dimension: {stats['dimension']} ")
        
        
    def embed_text(self, text:str) -> List[float]:
        """Generate embedding for a given text
        """
        embedding = self.embedder.encode(text).tolist()
        return embedding
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Batch embed all chunks in a single encode call
        Significantly faster than per-chunk encoding
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar = True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
    
    def upsert_chunks(self, chunks: List[Dict], namespace: str = 'default'):
        """
        Upset vectors into pinecone under a namespace
        Namespace isolates documents within the same index
        """
        # Implementation would depend on pinecone setup and chunk structure
        print("uploading to pinecone")
        
        vectors =[]
        for chunk in chunks:
            vectors.append(
                {
                    "id": chunk['chunk_id'],
                "values": chunk['embedding'],
                "metadata": {
                    "text": chunk['text'],
                    "source": chunk['source'],
                    "pages": chunk['pages'],
                    "word_count": chunk['word_count'],
                    "chunk_index": chunk['chunk_index']
                        
                    }
                }
            )
        #upsert to pinecone - in batches of 100
        batch_size = 100
        total = len(vectors)
        
        for i in range(0, total, batch_size):
            batch = vectors[i:i+batch_size]
            #pinecone upsert code here
            print(f"Upserting batch {i//batch_size + 1} with {len(batch)} vectors")
            self.index.upsert(vectors=batch, namespace=namespace)
            uploaded = min(i+batch_size, total)
            print(f"Upserted: {uploaded}/{total} chunks in pinecone")
    
    def search(
        self, 
        query:str, 
        top_k: int=3,
        namespace:str = "default",
        score_threshold: float = SCORE_THRESHOLD
        ) -> List[Dict]:
        """
        Retrieve top-K chunks above score threshold
        Return empty list if no matches meet the threshold
        """
        query_embedding = self.embed_text(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        matches = []
        for match in results['matches']:
            if match['score'] < score_threshold:
                continue
            matches.append({
                "chunk_id": match['id'],
                "score": match['score'],
                "text": match['metadata']['text'],
                "source": match['metadata']['source'],
                "pages": match['metadata'].get('pages', [])
            })
        print(f" found {len(matches)} matches")
        return matches
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in the index - use with caution
        """
        print("Deleting all vectors in the index")
        self.index.delete(delete_all=True, namespace=namespace)
        print("All vectors deleted")
        
    def get_stats(self) -> Dict:
        """Get index statistics
        """
        stats = self.index.describe_index_stats()
        return stats
    
if __name__ == "__main__":
    vs = VectorStore()
    
    test_text = "this is a test chunk of text to be embedded and stored in pinecone"
    embedding = vs.embed_text(test_text)
   
   
    dimensions_script = {len(embedding)}
    first_5_values = embedding[:5]
    results = vs.search("test_chunk", top_k=3)
    stats = vs.get_stats()
    
    with open("results.md","a") as f:
        f.write(f"\n ##new results \n")
        f.write(f"Dimesnions: {dimensions_script}")
        f.write(f"results= {results}")
        f.write(f"stats: {stats}")
        
        