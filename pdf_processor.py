import hashlib
import pdfplumber
from typing import List, Dict
import re

class PDFProcessor:
    """
    Extracts text from PDFs and chunks into digestible pieces
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"PDF processor initialized with chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    def extract_text(self, pdf_path:str) -> List[Dict]:
        """
        Extract text per page, preserving the page number for citation tracking
        Returns list of {page_num} dicts
        """
        pages = []
        print(f"extracting from path: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if text:
                    #normalize whitespace within page but preserve page boundary
                    text = re.sub(r'[ \t]+', ' ', text).strip()
                    pages.append({
                        'page_num': i+1,
                        'text': text
                    })
        return pages
    
    
    def chunk_text(self, pages: List[Dict], source: str) -> List[Dict]:
        """
        chunk word-by-word with overlap. Tracks page number and source.
        chunk_id is deterministic: hash of (source+chunk_index).
        """
        chunks = []
        #flatten all words  but track which page each word came from
        word_page_map = []
        for page in pages:
            words = page['text'].split()
            for word in words:
                word_page_map.append([word, page['page_num']])
        
        start = 0
        chunk_index = 0
    
        while start < len(word_page_map):
            end = min(start+ self.chunk_size, len(word_page_map))
            slice_ = word_page_map[start:end]
            
            chunk_words = [w for w, _ in slice_]
            chunk_pages = list(set(p for _, p in slice_))
            
            chunk_text_str = ' '.join(chunk_words)
            
            #deterministic ID:  prevents collision across documents
            raw_id = f"{source}_{chunk_index}"
            chunk_id = hashlib.md5(raw_id.encode()).hexdigest()
        
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text_str,
                'word_count': len(chunk_words),
                'source': source,
                'pages': sorted(chunk_pages),
                'chunk_index': chunk_index
            })
            
            chunk_index += 1
            
            start += self.chunk_size - self.chunk_overlap
        return chunks        

            
            
      
    def process_pdf(self,pdf_path: str) -> List[Dict]:
        """
        complete pipeline: extract+chunk
        """
        pages = self.extract_text(pdf_path)
        chunks = self.chunk_text(pages, source=pdf_path)
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
    