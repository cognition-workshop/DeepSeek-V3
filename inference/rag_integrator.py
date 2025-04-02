import os
from typing import List, Dict, Tuple, Optional
from pdf_processor import PDFProcessor
from vector_db import VectorDB

class RAGIntegrator:
    """
    Class for integrating RAG capabilities with DeepSeek's model.
    """
    def __init__(self, upload_dir: str = "uploads", index_dir: str = "indices"):
        """
        Initialize the RAG integrator.

        Args:
            upload_dir (str): Directory to store uploaded PDFs. Defaults to "uploads".
            index_dir (str): Directory to store vector indices. Defaults to "indices".
        """
        self.upload_dir = upload_dir
        self.index_dir = index_dir
        self.pdf_processor = PDFProcessor()
        self.vector_db = VectorDB()
        
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        
        self.current_pdf = None
        self.pdf_metadata = {}
        
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a PDF file and create a vector index.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        try:
            chunks, metadata = self.pdf_processor.process_pdf(pdf_path)
            self.current_pdf = pdf_path
            self.pdf_metadata = metadata
            
            self.vector_db.create_index(chunks)
            
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            index_path = os.path.join(self.index_dir, pdf_name)
            self.vector_db.save_index(index_path)
            
            return True
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False
            
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant context from the indexed PDF based on a query.

        Args:
            query (str): Query text.
            top_k (int): Number of chunks to retrieve. Defaults to 3.

        Returns:
            str: Concatenated context from relevant chunks.
        """
        results = self.vector_db.search(query, top_k)
        
        if not results:
            return ""
            
        context = "\n\n".join([chunk for _, chunk, _ in results])
        
        return context
