import torch
import torch.distributed as dist
from typing import Optional, List, Tuple
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Transformer, ModelArgs, ParallelEmbedding


class EmbeddingExtractor:
    """
    Extracts embeddings from DeepSeek-V3 ParallelEmbedding layer.
    
    Handles distributed embedding weights by aggregating across devices
    when world_size > 1, or using local weights for single-device setup.
    """
    
    def __init__(self, model: Transformer):
        self.model = model
        self.embedding_layer = model.embed
        
    def extract_embeddings(self, token_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Extract embeddings from the ParallelEmbedding layer.
        
        Args:
            token_ids: Optional list of specific token IDs to extract. If None, extracts all.
            
        Returns:
            Tuple of (embeddings_array, token_ids_list)
        """
        if not hasattr(self.embedding_layer, 'weight'):
            raise ValueError("Embedding layer does not have weight parameter")
            
        local_weights = self.embedding_layer.weight.data
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            all_weights = []
            for i in range(world_size):
                if i == rank:
                    all_weights.append(local_weights)
                else:
                    weight_shape = (self.embedding_layer.part_vocab_size, self.embedding_layer.dim)
                    all_weights.append(torch.zeros(weight_shape, dtype=local_weights.dtype, device=local_weights.device))
            
            for i, weight in enumerate(all_weights):
                if i != rank:
                    dist.broadcast(weight, src=i)
            
            full_weights = torch.cat(all_weights, dim=0)
        else:
            full_weights = local_weights
            
        embeddings = full_weights.cpu().float().numpy()
        
        if token_ids is not None:
            valid_token_ids = [tid for tid in token_ids if 0 <= tid < embeddings.shape[0]]
            if not valid_token_ids:
                raise ValueError("No valid token IDs provided")
            embeddings = embeddings[valid_token_ids]
            return embeddings, valid_token_ids
        else:
            all_token_ids = list(range(embeddings.shape[0]))
            return embeddings, all_token_ids
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding layer."""
        return {
            'vocab_size': self.embedding_layer.vocab_size,
            'embedding_dim': self.embedding_layer.dim,
            'part_vocab_size': getattr(self.embedding_layer, 'part_vocab_size', self.embedding_layer.vocab_size),
            'vocab_start_idx': getattr(self.embedding_layer, 'vocab_start_idx', 0),
            'vocab_end_idx': getattr(self.embedding_layer, 'vocab_end_idx', self.embedding_layer.vocab_size),
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'rank': dist.get_rank() if dist.is_initialized() else 0
        }
