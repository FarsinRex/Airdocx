import os
from typing import List, Dict
from dotenv import load_dotenv
import pinecone

from sentence_transformers import SentenceTransformer
load_dotenv()

class VectorStore:
    """Manages embeddings and pinecone storage
    """
    def __init__(self):
        #initliaze embedding model 
        print("Loading embedding model")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        print("connecting to pinecone")
        
        
        #initiaze pinecone connection
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment='us-east-1-aws'
        )
        self.index_name = os.getenv('PINECONE_INDEX_NAME')
        self.index_host = os.getenv('PINECONE_HOST')
        self.index = pinecone.Index(index_name=self.index_name, host=sectvilf.index_host)
        
        print(f"    Connected to index: {self.index_name}")
        
    def embed_text(self, text:str) -> List[float]:
        """Generate embedding for a given text
        """
        embedding = self.embedder.encode(text)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for a list of text chunks
        """
        print(f"Embedding {len(chunks)} chunks")
        for chunk in chunks:
            chunk['embedding'] = self.embed_text(chunk['text'])
        print("All chunks embedded")
        return chunks
    
    def upsert_chunks(self, chunks: List[Dict], source: str = "unknown.pdf"):
        """store chunks in pinecone
        """
        # Implementation would depend on pinecone setup and chunk structure
        print("uploading to pinecone")
        
        vectors =[]
        for chunk in chunks:
            vectors.append(
                {
                    "id": chunk['id'],
                    "values": chunk['embedding'],
                    "metadata": {
                        "text": chunk['text'],
                        'source': source,
                        'word_count': chunk['word_count']
                        
                    }
                }
            )
        #upsert to pinecone - in batches of 100
        batch_size = 100
        
        for i in range(0, len(vectors),batch_size):
            batch = vectors[i:i+batch_size]
            #pinecone upsert code here
            print(f"Upserting batch {i//batch_size + 1} with {len(batch)} vectors")
            self.index.upsert(vectors=batch)
            print(f" uploade {min(i+batch_size, len(vectors))/len(vectors)}")
        print(f"stored: {len(vectors)} chunks in pinecone")
    
    def search(self, query:str, top_k: int=3) -> List[Dict]:
        """
        Search for similar chunks in pinecone based on a query
        """
        query_embedding = self.embed_text(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results['matches']:
            matches.append({
                "chunk_id": match['id'],
                "score": match['score'],
                "text": match['metadata']['text'],
                "source": match['metadata']['source']
            })
        print(f" found {len(matches)} matches")
        return matches
    
    def delete_all(self):
        """Delete all vectors in the index - use with caution
        """
        print("Deleting all vectors in the index")
        self.index.delete(delete_all=True)
        print("All vectors deleted")
        
    def get_stats(self):
        """Get index statistics
        """
        stats = self.index.describe_index_stats()
        return stats
    
if __name__ == "__main__":
    vs = VectorStore()
    
    test_text = "this is a test chunk of text to be embedded and stored in pinecone"
    embedding = vs.embed_text(test_text)
    print(f"test embedding: {len(embedding)} dimensions")
    print(f" first 5 values: {embedding[:5]}")
    
    results = vs.search("test_chunk")
    
    
    results = vs.search("test_chunk", top_k=3)
    print(f"Search results: {len(results)} results")
    
    stats = vs.get_stats()
    print(f"Index stats: {stats}")