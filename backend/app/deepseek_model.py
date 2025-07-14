import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
import json
import os
from dataclasses import dataclass

world_size = 1
rank = 0

@dataclass
class ModelArgs:
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 3
    n_heads: int = 128
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 2.5
    score_func: str = "sigmoid"
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    dtype: str = "fp8"
    max_seq_len: int = 4096
    max_batch_size: int = 32

class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.
    Simplified version for visualization purposes.
    """
    def __init__(self, vocab_size: int, dim: int, world_size: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.world_size = world_size
        
        if vocab_size % world_size != 0:
            padded_vocab_size = ((vocab_size + world_size - 1) // world_size) * world_size
            self.vocab_size = padded_vocab_size
        
        self.part_vocab_size = self.vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        
        self.weight = nn.Parameter(torch.randn(self.vocab_size, self.dim) * 0.02)
        
    def forward(self, x: torch.Tensor, simulate_distributed: bool = False) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with optional distributed simulation for visualization.
        
        Returns:
            embeddings: The embedding tensor
            debug_info: Dictionary with distributed processing information
        """
        debug_info = {
            "world_size": self.world_size,
            "vocab_size": self.vocab_size,
            "part_vocab_size": self.part_vocab_size,
            "vocab_start_idx": self.vocab_start_idx,
            "vocab_end_idx": self.vocab_end_idx,
            "input_shape": x.shape,
            "masked_tokens": [],
            "partition_info": {}
        }
        
        if simulate_distributed and self.world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            debug_info["masked_tokens"] = x[mask].tolist() if mask.any() else []
            
            x_local = x - self.vocab_start_idx
            x_local[mask] = 0
            
            y = F.embedding(x_local, self.weight[self.vocab_start_idx:self.vocab_end_idx])
            
            y[mask] = 0
            
            debug_info["partition_info"] = {
                "local_vocab_range": [self.vocab_start_idx, self.vocab_end_idx],
                "masked_positions": mask.nonzero().squeeze().tolist() if mask.any() else []
            }
        else:
            y = F.embedding(x, self.weight)
        
        return y, debug_info

    def get_partition_visualization_data(self):
        """Get data for visualizing vocabulary partitioning."""
        partitions = []
        for i in range(self.world_size):
            start_idx = i * self.part_vocab_size
            end_idx = start_idx + self.part_vocab_size
            partitions.append({
                "rank": i,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "size": self.part_vocab_size,
                "percentage": (self.part_vocab_size / self.vocab_size) * 100
            })
        return partitions

def create_mock_deepseek_model(config_path: Optional[str] = None, world_size: int = 1):
    """Create a simplified DeepSeek-V3 model for embedding visualization."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config_dict = json.load(f)
        args = ModelArgs(**config_dict)
    else:
        args = ModelArgs()
    
    embedding_layer = ParallelEmbedding(args.vocab_size, args.dim, world_size)
    
    return embedding_layer, args
