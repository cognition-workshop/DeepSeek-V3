#!/usr/bin/env python3
"""Test script to verify embedding extraction and visualization functionality."""

import sys
import os
import torch
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.embedding_extractor import EmbeddingExtractor
from visualization.dimensionality_reduction import DimensionalityReducer
from model import Transformer, ModelArgs

def test_embedding_extraction():
    """Test embedding extraction with a mock model."""
    print("Testing embedding extraction...")
    
    with open('configs/config_16B.json') as f:
        args = ModelArgs(**json.load(f))
    
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")  # Use CPU for testing
    
    with torch.device("cpu"):
        model = Transformer(args)
    
    extractor = EmbeddingExtractor(model)
    info = extractor.get_embedding_info()
    
    print(f"✓ Embedding info: vocab_size={info['vocab_size']}, dim={info['embedding_dim']}")
    
    try:
        embeddings, token_ids = extractor.extract_embeddings(token_ids=list(range(100)))
        print(f"✓ Extracted embeddings shape: {embeddings.shape}")
        print(f"✓ Token IDs count: {len(token_ids)}")
        return embeddings, token_ids
    except Exception as e:
        print(f"✗ Embedding extraction failed: {e}")
        return None, None

def test_dimensionality_reduction(embeddings):
    """Test dimensionality reduction methods."""
    if embeddings is None:
        print("Skipping dimensionality reduction test - no embeddings")
        return
        
    print("\nTesting dimensionality reduction...")
    reducer = DimensionalityReducer()
    
    try:
        reduced_pca = reducer.reduce_pca(embeddings, n_components=2)
        print(f"✓ PCA reduction successful: {reduced_pca.shape}")
    except Exception as e:
        print(f"✗ PCA failed: {e}")
    
    try:
        reduced_tsne = reducer.reduce_tsne(embeddings, n_components=2, perplexity=min(30, len(embeddings)//3))
        print(f"✓ t-SNE reduction successful: {reduced_tsne.shape}")
    except Exception as e:
        print(f"✗ t-SNE failed: {e}")
    
    try:
        reduced_umap = reducer.reduce_umap(embeddings, n_components=2)
        print(f"✓ UMAP reduction successful: {reduced_umap.shape}")
    except Exception as e:
        print(f"✗ UMAP failed: {e}")

def main():
    """Run all functionality tests."""
    print("=== DeepSeek-V3 Embedding Visualization Functionality Test ===\n")
    
    embeddings, token_ids = test_embedding_extraction()
    test_dimensionality_reduction(embeddings)
    
    print("\n=== Test Complete ===")
    print("If all tests passed, the web application should work correctly.")
    print("Run: python visualize_embeddings.py --ckpt-path /path/to/model --config configs/config_16B.json")

if __name__ == "__main__":
    main()
