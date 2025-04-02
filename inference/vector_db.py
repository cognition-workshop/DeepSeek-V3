import os
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

class VectorDB:
    """
    Vector database for storing and retrieving text embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector database.

        Args:
            model_name (str): Name of the SentenceTransformer model to use for embeddings.
                             Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.embeddings = None
        
    def create_index(self, chunks: List[str]) -> None:
        """
        Create a FAISS index from text chunks.

        Args:
            chunks (List[str]): List of text chunks to index.
        """
        self.chunks = chunks
        
        self.embeddings = self.model.encode(chunks)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        Search for similar chunks to a query.

        Args:
            query (str): Query text.
            top_k (int): Number of results to return. Defaults to 3.

        Returns:
            List[Tuple[int, str, float]]: List of tuples containing (chunk_index, chunk_text, similarity_score).
        """
        if self.index is None or not self.chunks:
            return []
        
        query_embedding = self.model.encode([query])
        
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks) and idx >= 0:  # Valid index check
                results.append((int(idx), self.chunks[idx], float(distances[0][i])))
                
        return results
    
    def save_index(self, path: str) -> None:
        """
        Save the FAISS index and chunks to disk.

        Args:
            path (str): Directory path to save the index.
        """
        if self.index is None:
            return
            
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        if self.embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "chunks.txt"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk + "\n===CHUNK_SEPARATOR===\n")
                
    def load_index(self, path: str) -> bool:
        """
        Load a FAISS index and chunks from disk.

        Args:
            path (str): Directory path to load the index from.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            self.index = faiss.read_index(os.path.join(path, "index.faiss"))
            self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
            
            with open(os.path.join(path, "chunks.txt"), "r", encoding="utf-8") as f:
                content = f.read()
                self.chunks = content.split("\n===CHUNK_SEPARATOR===\n")[:-1]  # Remove last empty element
                
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
