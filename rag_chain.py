import os
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
from vector_store import VectorStore

load_dotenv()

class RAGChain:
    def __init__(self, 
                 vector_store: VectorStore,
                 namespace: str = "default",
                 top_k: int =5,
                 score_threshold: float = 0.72
    ):
        
        
        self.vector_store = vector_store
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = 'llama-3.3-70b-versatile'
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.namespace = namespace

    def retrieve_context(self, query: str) -> List[Dict]:
        return self.vector_store.search(
            query,
            top_k=self.top_k,
            namespace=self.namespace,
            score_threshold= self.score_threshold
        )

    def build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        context = "\n\n".join([
            f"[Source {i+1} | {chunk['source']} | Pages {chunk['pages']}]\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        return f"""You are a precise assistant that answers questions strictly from the provided document context.
Rules:
- Use ONLY the context below. Do not use prior knowledge.
- If the answer is not in the context, respond exactly: "I cannot find this in the provided document."
- Cite the source number(s) you used in your answer.

Context:
{context}

Question: {question}

Answer:"""

    def answer(self, question: str) -> Dict:
        context_chunks = self.retrieve_context(question)

        if not context_chunks:
            return {
                'answer': 'No relevant content found above confidence threshold. The document may not contain information about this topic.',
                'sources': [],
                'context_used': [],
                'model': self.model
            }

        prompt = self.build_prompt(question, context_chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # low temperature = less creative = less hallucination
            max_tokens=1024
        )

        answer_text = response.choices[0].message.content.strip()

        return {
            'answer': answer_text,
            'sources': list(set(c['source'] for c in context_chunks)),
            'context_used': [
                {'chunk_id': c['chunk_id'], 'score': c['score'], 'pages': c['pages']}
                for c in context_chunks
            ],
            'model': self.model
        }