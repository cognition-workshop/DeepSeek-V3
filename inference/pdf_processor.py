import os
import PyPDF2
from typing import List, Dict, Tuple

class PDFProcessor:
    """
    Class for processing PDF documents, extracting text, and chunking content.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor.

        Args:
            chunk_size (int): Size of text chunks in characters. Defaults to 1000.
            chunk_overlap (int): Overlap between chunks in characters. Defaults to 200.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pdf_path = None
        self.chunks = []
        self.pdf_metadata = {}

    def process_pdf(self, pdf_path: str) -> Tuple[List[str], Dict]:
        """
        Process a PDF file, extract text, and create chunks.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            Tuple[List[str], Dict]: A tuple containing a list of text chunks and PDF metadata.
        """
        self.pdf_path = pdf_path
        self.chunks = []
        self.pdf_metadata = {}
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            self.pdf_metadata = {
                'title': os.path.basename(pdf_path),
                'pages': len(reader.pages),
                'filename': os.path.basename(pdf_path)
            }
            
            full_text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                full_text += page.extract_text() + " "
        
        self.chunks = self._create_chunks(full_text)
        
        return self.chunks, self.pdf_metadata
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text (str): Full text to be chunked.

        Returns:
            List[str]: List of text chunks.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            
        return chunks
