from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import json
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer

from .deepseek_model import create_mock_deepseek_model, ModelArgs

app = FastAPI()

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = None
tokenizer = None
model_config = None

class TextInput(BaseModel):
    text: str
    world_size: Optional[int] = 1

class EmbeddingResponse(BaseModel):
    tokens: List[str]
    token_ids: List[int]
    embeddings: List[List[float]]
    vocab_partitions: Dict[str, Any]
    visualization_data: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "DeepSeek-V3 Embedding Visualizer API", "status": "running"}

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/initialize")
def initialize_model(world_size: int = 1):
    """Initialize the DeepSeek-V3 embedding model and tokenizer."""
    global model, tokenizer, model_config
    
    try:
        model, model_config = create_mock_deepseek_model(world_size=world_size)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        return {
            "status": "success",
            "message": "Model and tokenizer initialized successfully",
            "config": {
                "vocab_size": model_config.vocab_size,
                "dim": model_config.dim,
                "world_size": world_size
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")

@app.post("/embed", response_model=EmbeddingResponse)
def get_embeddings(input_data: TextInput):
    """Get embeddings for input text with distributed processing visualization."""
    global model, tokenizer, model_config
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Call /initialize first.")
    
    try:
        world_size = input_data.world_size or 1
        
        if model.world_size != world_size:
            model, model_config = create_mock_deepseek_model(world_size=world_size)
        
        tokens = tokenizer.tokenize(input_data.text)
        token_ids = tokenizer.encode(input_data.text, return_tensors="pt")
        
        with torch.no_grad():
            embeddings, debug_info = model(token_ids, simulate_distributed=world_size > 1)
        
        embeddings_list = embeddings.squeeze(0).tolist()
        
        vocab_partitions = {
            "partitions": model.get_partition_visualization_data(),
            "debug_info": debug_info
        }
        
        visualization_data = create_visualization_data(embeddings.squeeze(0), tokens, token_ids.squeeze(0))
        
        return EmbeddingResponse(
            tokens=tokens,
            token_ids=token_ids.squeeze(0).tolist(),
            embeddings=embeddings_list,
            vocab_partitions=vocab_partitions,
            visualization_data=visualization_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get embeddings: {str(e)}")

def create_visualization_data(embeddings: torch.Tensor, tokens: List[str], token_ids: torch.Tensor) -> Dict[str, Any]:
    """Create data for various visualizations."""
    embeddings_np = embeddings.numpy()
    
    if embeddings_np.shape[0] > 1:
        pca = PCA(n_components=min(3, embeddings_np.shape[0]))
        pca_result = pca.fit_transform(embeddings_np)
        
        pca_data = {
            "x": pca_result[:, 0].tolist(),
            "y": pca_result[:, 1].tolist(),
            "z": pca_result[:, 2].tolist() if pca_result.shape[1] > 2 else [0] * len(pca_result),
            "tokens": tokens,
            "token_ids": token_ids.tolist(),
            "explained_variance": pca.explained_variance_ratio_.tolist()
        }
    else:
        pca_data = {
            "x": [0], "y": [0], "z": [0],
            "tokens": tokens, "token_ids": token_ids.tolist(),
            "explained_variance": [1.0]
        }
    
    tsne_data = None
    if embeddings_np.shape[0] > 3:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings_np.shape[0]-1))
            tsne_result = tsne.fit_transform(embeddings_np)
            tsne_data = {
                "x": tsne_result[:, 0].tolist(),
                "y": tsne_result[:, 1].tolist(),
                "tokens": tokens,
                "token_ids": token_ids.tolist()
            }
        except:
            tsne_data = None
    
    similarity_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    
    return {
        "pca": pca_data,
        "tsne": tsne_data,
        "similarity_matrix": similarity_matrix.tolist(),
        "embedding_stats": {
            "mean": embeddings_np.mean(axis=1).tolist(),
            "std": embeddings_np.std(axis=1).tolist(),
            "norm": torch.norm(embeddings, dim=1).tolist()
        }
    }

@app.get("/model-info")
def get_model_info():
    """Get information about the current model configuration."""
    global model, model_config
    
    if model is None or model_config is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    
    return {
        "config": {
            "vocab_size": model_config.vocab_size,
            "dim": model_config.dim,
            "world_size": model.world_size,
            "part_vocab_size": model.part_vocab_size
        },
        "partition_info": model.get_partition_visualization_data()
    }
