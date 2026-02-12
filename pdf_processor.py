import PyPDF2
from typing import List, Dict
import re

class PDFProcessor:
    """
    Extracts text from PDFs and chunks into digestible pieces
    """
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"PDF processor initialized with chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    def extract_text(self, pdf_path:str) -> str:
        """
        Extract all the text from a PDF file
        """
        print(f"extracting from path: {pdf_path}")
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            print(f"  pages: {num_pages}")
            for page in reader.pages:
                #calling extract_text from PyPDF2 on each page and concatenating results
                text+= page.extract_text() + "\n"
        #removing extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"  extracted {len(text)} characters")
        return text
    def chunk_text(self, text:str) -> List[Dict]:
        """Split text into overlapping chunks
        """
        #positive lookbehind to split on sentence boundaries while keeping punctuation
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunks = []
        current_chunk = []
        current_words = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            
            if current_words + words > self.chunk_size and current_chunk:
                 
                 chunk_text = ''.join(current_chunk)
                 chunks.append(
                     {
                         'chunk_id': f"chunk_{len(chunks)}",
                         'text': chunk_text,
                         'word_count': current_words
                     }
                 )
            else:
                current_chunk.append(sentence)
                current_words += words
        #final_chunk
        if current_chunk:
            chunks.append({
                'chunk_id': f"chunk_{len(chunks)}",
                'text': ''.join(current_chunk),
                'word_count': current_words
            })
            
        print(f"created {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self,pdf_path: str) -> List[Dict]:
        """
        complete pipeline: extract+chunk
        """
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        return chunks

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Error: you must provide only one path to a PDF file")
        print("Usage : python pdf_processor.py <pdf_path>")
        sys.exit(1)
        
    processor = PDFProcessor()
    chunks = processor.process_pdf(sys.argv[1])
    
    print(f"\n Summary: {len(chunks)}")
    print(f"Chunks created: {len(chunks)}", chunks)
    