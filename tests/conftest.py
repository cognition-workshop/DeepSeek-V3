import pytest
import torch
import torch.distributed as dist
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any
import json
import os

from inference.model import ModelArgs


@pytest.fixture
def mock_distributed():
    with patch('torch.distributed.is_initialized', return_value=False), \
         patch('torch.distributed.get_world_size', return_value=1), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.all_reduce') as mock_all_reduce, \
         patch('torch.distributed.all_gather') as mock_all_gather:
        
        mock_all_reduce.return_value = None
        mock_all_gather.return_value = None
        yield {
            'all_reduce': mock_all_reduce,
            'all_gather': mock_all_gather
        }


@pytest.fixture
def mock_kernels():
    def mock_act_quant(x, block_size=128):
        y = x.to(torch.float8_e4m3fn)
        s = torch.ones(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32, device=x.device)
        return y, s
    
    def mock_weight_dequant(x, s, block_size=128):
        return x.to(torch.get_default_dtype())
    
    def mock_fp8_gemm(a, a_s, b, b_s):
        return torch.matmul(a.to(torch.float32), b.to(torch.float32).T)
    
    def mock_linear(x, weight, bias=None):
        x_compute = x.to(torch.float32)
        weight_compute = weight.to(torch.float32)
        
        if torch.isnan(weight_compute).any() or torch.isinf(weight_compute).any():
            weight_compute = torch.randn_like(weight_compute) * 0.1
        
        result = torch.matmul(x_compute, weight_compute.T)
        
        if bias is not None:
            bias_compute = bias.to(torch.float32)
            if torch.isnan(bias_compute).any() or torch.isinf(bias_compute).any():
                bias_compute = torch.zeros_like(bias_compute)
            result = result + bias_compute
        
        result = torch.clamp(result, -10.0, 10.0)
        
        return result.to(x.dtype)
    
    with patch('inference.kernel.act_quant', side_effect=mock_act_quant), \
         patch('inference.kernel.weight_dequant', side_effect=mock_weight_dequant), \
         patch('inference.kernel.fp8_gemm', side_effect=mock_fp8_gemm), \
         patch('inference.model.linear', side_effect=mock_linear):
        yield


@pytest.fixture
def config_16b():
    return ModelArgs(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=27,
        n_dense_layers=1,
        n_heads=16,
        n_routed_experts=64,
        n_shared_experts=2,
        n_activated_experts=6,
        route_scale=1.0,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=0.707,
        dtype="bf16"
    )


@pytest.fixture
def config_236b():
    return ModelArgs(
        vocab_size=102400,
        dim=5120,
        inter_dim=12288,
        moe_inter_dim=1536,
        n_layers=60,
        n_dense_layers=1,
        n_heads=128,
        n_routed_experts=160,
        n_shared_experts=2,
        n_activated_experts=6,
        n_expert_groups=8,
        n_limited_groups=3,
        route_scale=16.0,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dtype="bf16"
    )


@pytest.fixture
def config_671b():
    return ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=8,
        n_expert_groups=8,
        n_limited_groups=4,
        route_scale=2.5,
        score_func="sigmoid",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dtype="fp8"
    )


@pytest.fixture
def simple_config():
    return ModelArgs(
        dim=512,
        moe_inter_dim=256,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        route_scale=1.0,
        score_func="softmax",
        dtype="bf16"
    )


@pytest.fixture(autouse=True)
def setup_torch():
    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")
    yield
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")


@pytest.fixture
def sample_input():
    def _create_input(batch_size=2, seq_len=4, dim=512):
        return torch.randn(batch_size, seq_len, dim)
    return _create_input
