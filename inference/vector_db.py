import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModel, AutoTokenizer

class VectorDB:
    """
    Simple vector database for storing and retrieving text embeddings.
    Uses cosine similarity for retrieval without requiring external dependencies.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the vector database.

        Args:
            model_name (str): Name of the model to use for embeddings.
                             Defaults to "bert-base-uncased".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunks = []
        self.embeddings = None
        
    def create_index(self, chunks: List[str]) -> None:
        """
        Create embeddings for text chunks.

        Args:
            chunks (List[str]): List of text chunks to index.
        """
        self.chunks = chunks
        
        embeddings = []
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        self.embeddings = np.array(embeddings)
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, str, float]]:
        """
        Search for similar chunks to a query using cosine similarity.

        Args:
            query (str): Query text.
            top_k (int): Number of results to return. Defaults to 3.

        Returns:
            List[Tuple[int, str, float]]: List of tuples containing (chunk_index, chunk_text, similarity_score).
        """
        if self.embeddings is None or not self.chunks:
            return []
        
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            query_norm = np.linalg.norm(query_embedding)
            embedding_norm = np.linalg.norm(embedding)
            
            if query_norm > 0 and embedding_norm > 0:
                similarity = np.dot(query_embedding, embedding) / (query_norm * embedding_norm)
                similarities.append((i, similarity))
            else:
                similarities.append((i, 0.0))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(top_k, len(similarities))):
            idx, score = similarities[i]
            results.append((idx, self.chunks[idx], float(score)))
                
        return results
    
    def save_index(self, path: str) -> None:
        """
        Save chunks and embeddings to disk.

        Args:
            path (str): Directory path to save the data.
        """
        if self.embeddings is None:
            return
            
        os.makedirs(path, exist_ok=True)
        
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "chunks.txt"), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk + "\n===CHUNK_SEPARATOR===\n")
                
    def load_index(self, path: str) -> bool:
        """
        Load chunks and embeddings from disk.

        Args:
            path (str): Directory path to load the data from.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
            
            with open(os.path.join(path, "chunks.txt"), "r", encoding="utf-8") as f:
                content = f.read()
                self.chunks = content.split("\n===CHUNK_SEPARATOR===\n")[:-1]  # Remove last empty element
                
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
