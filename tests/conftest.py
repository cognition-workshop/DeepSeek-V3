import pytest
import torch
import torch.distributed as dist
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

@pytest.fixture(autouse=True)
def mock_distributed():
    with patch('torch.distributed.is_initialized', return_value=False), \
         patch('torch.distributed.get_world_size', return_value=1), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.all_reduce'), \
         patch('torch.distributed.all_gather'):
        yield

@pytest.fixture(autouse=True)
def mock_triton_kernels():
    def mock_act_quant(x, block_size=128):
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
        s.fill_(1.0)
        return y, s
    
    def mock_weight_dequant(x, s, block_size=128):
        return torch.empty_like(x, dtype=torch.get_default_dtype())
    
    def mock_fp8_gemm(a, a_s, b, b_s):
        K = a.size(-1)
        M = a.numel() // K
        N = b.size(0)
        return a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    
    with patch('kernel.act_quant', side_effect=mock_act_quant), \
         patch('kernel.weight_dequant', side_effect=mock_weight_dequant), \
         patch('kernel.fp8_gemm', side_effect=mock_fp8_gemm):
        yield

@pytest.fixture
def device():
    return torch.device('cpu')

@pytest.fixture
def dtype():
    return torch.bfloat16

@pytest.fixture
def model_args():
    from model import ModelArgs
    torch.set_default_dtype(torch.bfloat16)
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        vocab_size=1000,
        dim=512,
        inter_dim=1024,
        moe_inter_dim=256,
        n_layers=4,
        n_heads=8,
        n_routed_experts=16,
        n_shared_experts=2,
        n_activated_experts=2
    )

@pytest.fixture
def small_model_args():
    from model import ModelArgs
    torch.set_default_dtype(torch.bfloat16)
    return ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        vocab_size=100,
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        n_layers=2,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2
    )

@pytest.fixture
def sample_tokens():
    return torch.randint(0, 100, (2, 16))

@pytest.fixture
def sample_embeddings():
    return torch.randn(2, 16, 64, dtype=torch.bfloat16)
