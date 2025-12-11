"""
Embedding Visualization Backend

This FastAPI application provides endpoints for:
1. Uploading PDF files
2. Extracting text from PDFs
3. Generating embeddings using a simplified embedding approach inspired by DeepSeek-V3's ParallelEmbedding
4. Performing dimensionality reduction (t-SNE/PCA) for visualization
"""

import io
import sys
import os
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add the inference directory to path to use the model infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'inference'))

app = FastAPI(
    title="DeepSeek-V3 Embedding Visualizer",
    description="Visualize embeddings extracted from PDF files using DeepSeek-V3 inspired architecture",
    version="1.0.0"
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration inspired by DeepSeek-V3's ModelArgs
@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model, inspired by DeepSeek-V3's ModelArgs."""
    vocab_size: int = 102400
    dim: int = 2048  # Same as DeepSeek-V3's embedding dimension
    max_seq_len: int = 4096


# Global embedding model instance
embedding_model = None


class SimpleEmbedding:
    """
    A simplified embedding layer inspired by DeepSeek-V3's ParallelEmbedding class.
    
    This implementation creates embeddings without requiring the full model weights,
    using a deterministic hash-based approach for demonstration purposes.
    
    The original ParallelEmbedding (inference/model.py lines 87-126) handles
    vocabulary partitioning across distributed processes. This simplified version
    maintains the same embedding dimension (2048) for compatibility.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        
        # Initialize a random embedding matrix with fixed seed for reproducibility
        np.random.seed(42)
        self.embedding_matrix = np.random.randn(config.vocab_size, config.dim).astype(np.float32)
        # Normalize embeddings
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        self.embedding_matrix = self.embedding_matrix / norms
        
    def tokenize(self, text: str) -> List[int]:
        """
        Simple character-level tokenization.
        Maps each character to a token ID within the vocabulary range.
        """
        tokens = []
        for char in text:
            # Use character ordinal modulo vocab_size to get token ID
            token_id = ord(char) % self.vocab_size
            tokens.append(token_id)
        return tokens
    
    def embed(self, tokens: List[int]) -> np.ndarray:
        """
        Convert token IDs to embeddings.
        
        Similar to ParallelEmbedding.forward() which uses F.embedding(x, self.weight)
        to convert token indices to dense vectors.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            numpy array of shape (seq_len, dim) containing embeddings
        """
        if not tokens:
            return np.zeros((1, self.dim), dtype=np.float32)
        
        # Clip token IDs to valid range
        tokens = [min(max(t, 0), self.vocab_size - 1) for t in tokens]
        
        # Look up embeddings (similar to F.embedding in PyTorch)
        embeddings = self.embedding_matrix[tokens]
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string by tokenizing and then embedding.
        
        Returns the mean embedding across all tokens (pooling).
        """
        tokens = self.tokenize(text)
        embeddings = self.embed(tokens)
        # Mean pooling to get a single embedding vector
        return np.mean(embeddings, axis=0)
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Embed multiple text chunks.
        
        Args:
            chunks: List of text strings
            
        Returns:
            numpy array of shape (n_chunks, dim)
        """
        embeddings = []
        for chunk in chunks:
            emb = self.embed_text(chunk)
            embeddings.append(emb)
        return np.array(embeddings)


def get_embedding_model() -> SimpleEmbedding:
    """Get or create the embedding model instance."""
    global embedding_model
    if embedding_model is None:
        config = EmbeddingConfig()
        embedding_model = SimpleEmbedding(config)
    return embedding_model


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    text_chunks: List[str]
    reduction_method: str = "pca"  # "pca" or "tsne"
    n_components: int = 2  # 2 or 3 for visualization


class EmbeddingResponse(BaseModel):
    """Response model for embedding visualization data."""
    reduced_embeddings: List[List[float]]
    original_dim: int
    n_chunks: int
    text_chunks: List[str]
    reduction_method: str


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload."""
    text_chunks: List[str]
    total_pages: int
    total_characters: int


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeepSeek-V3 Embedding Visualizer API",
        "docs": "/docs",
        "endpoints": {
            "/upload-pdf": "Upload a PDF file and extract text",
            "/generate-embeddings": "Generate embeddings from text chunks",
            "/health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "embedding_dim": EmbeddingConfig.dim}


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    overlap: int = 50
):
    """
    Upload a PDF file and extract text content.
    
    Args:
        file: The PDF file to upload
        chunk_size: Size of each text chunk in characters
        overlap: Overlap between consecutive chunks
        
    Returns:
        Extracted text chunks and metadata
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF content
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        # Extract text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Split into chunks with overlap
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
            if end >= len(full_text):
                break
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [full_text.strip()]
        
        return PDFUploadResponse(
            text_chunks=chunks,
            total_pages=len(pdf_reader.pages),
            total_characters=len(full_text)
        )
        
    except PyPDF2.errors.PdfReadError as e:
        raise HTTPException(status_code=400, detail=f"Invalid PDF file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/generate-embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for text chunks and reduce dimensionality for visualization.
    
    This endpoint uses a simplified embedding approach inspired by DeepSeek-V3's
    ParallelEmbedding class (inference/model.py lines 87-126).
    
    The embeddings have dimension 2048 (matching DeepSeek-V3's config at line 58),
    which are then reduced to 2D or 3D using PCA or t-SNE for visualization.
    
    Args:
        request: Contains text chunks and visualization parameters
        
    Returns:
        Reduced embeddings suitable for 2D/3D visualization
    """
    if not request.text_chunks:
        raise HTTPException(status_code=400, detail="No text chunks provided")
    
    if request.n_components not in [2, 3]:
        raise HTTPException(status_code=400, detail="n_components must be 2 or 3")
    
    if request.reduction_method not in ["pca", "tsne"]:
        raise HTTPException(status_code=400, detail="reduction_method must be 'pca' or 'tsne'")
    
    try:
        # Get embedding model
        model = get_embedding_model()
        
        # Generate embeddings for all chunks
        embeddings = model.embed_chunks(request.text_chunks)
        
        # Apply dimensionality reduction
        n_samples = len(request.text_chunks)
        
        if request.reduction_method == "tsne":
            # t-SNE requires at least n_components + 1 samples
            # and perplexity must be less than n_samples
            if n_samples < request.n_components + 1:
                # Fall back to PCA for small datasets
                reducer = PCA(n_components=request.n_components)
            else:
                perplexity = min(30, max(5, n_samples - 1))
                reducer = TSNE(
                    n_components=request.n_components,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=1000
                )
        else:
            reducer = PCA(n_components=request.n_components)
        
        # Reduce dimensionality
        reduced = reducer.fit_transform(embeddings)
        
        return EmbeddingResponse(
            reduced_embeddings=reduced.tolist(),
            original_dim=model.dim,
            n_chunks=n_samples,
            text_chunks=request.text_chunks,
            reduction_method=request.reduction_method
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """
    Get information about the embedding model configuration.
    
    Returns configuration details inspired by DeepSeek-V3's ModelArgs.
    """
    config = EmbeddingConfig()
    return {
        "model_name": "DeepSeek-V3 Inspired Embedding",
        "vocab_size": config.vocab_size,
        "embedding_dim": config.dim,
        "max_seq_len": config.max_seq_len,
        "description": "Simplified embedding layer inspired by DeepSeek-V3's ParallelEmbedding class",
        "reference": "inference/model.py lines 87-126"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
