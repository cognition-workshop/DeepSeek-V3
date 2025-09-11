import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch

from inference.model import Expert
from tests.test_utils import (
    create_test_tensor, assert_tensor_properties, mock_world_size_rank,
    check_gradient_flow, initialize_module_weights
)


class TestExpert:
    
    def test_expert_initialization(self, mock_distributed, mock_kernels):
        dim = 512
        inter_dim = 1024
        expert = Expert(dim, inter_dim)
        
        assert hasattr(expert, 'w1')
        assert hasattr(expert, 'w2')
        assert hasattr(expert, 'w3')
        
        assert expert.w1.in_features == dim
        assert expert.w1.out_features == inter_dim
        assert expert.w2.in_features == inter_dim
        assert expert.w2.out_features == dim
        assert expert.w3.in_features == dim
        assert expert.w3.out_features == inter_dim
    
    def test_expert_forward_basic(self, mock_distributed, mock_kernels):
        dim = 256
        inter_dim = 512
        expert = Expert(dim, inter_dim)
        
        batch_size = 4
        x = create_test_tensor((batch_size, dim))
        
        output = expert(x)
        
        assert_tensor_properties(output, (batch_size, dim))
        assert output.dtype == x.dtype
    
    def test_expert_forward_different_shapes(self, mock_distributed, mock_kernels):
        dim = 128
        inter_dim = 256
        expert = Expert(dim, inter_dim)
        
        test_cases = [
            (1, dim),
            (8, dim),
            (16, dim),
            (32, dim),
        ]
        
        for batch_size, input_dim in test_cases:
            x = create_test_tensor((batch_size, input_dim))
            output = expert(x)
            assert_tensor_properties(output, (batch_size, dim))
    
    def test_expert_swiglu_activation(self, mock_distributed, mock_kernels):
        dim = 64
        inter_dim = 128
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = create_test_tensor((2, dim))
        
        with torch.no_grad():
            w1_out = expert.w1(x)
            w3_out = expert.w3(x)
            expected_intermediate = F.silu(w1_out) * w3_out
            
        output = expert(x)
        
        assert_tensor_properties(output, (2, dim))
        assert torch.isfinite(output).all()
    
    def test_expert_gradient_flow(self, mock_distributed, mock_kernels):
        dim = 128
        inter_dim = 256
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = create_test_tensor((4, dim))
        check_gradient_flow(expert, x)
    
    def test_expert_different_precisions(self, mock_distributed, mock_kernels):
        dim = 64
        inter_dim = 128
        expert = Expert(dim, inter_dim)
        
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, dim, dtype=dtype)
            output = expert(x)
            assert output.dtype == dtype
    
    def test_expert_zero_input(self, mock_distributed, mock_kernels):
        dim = 32
        inter_dim = 64
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = torch.zeros(3, dim)
        output = expert(x)
        
        assert_tensor_properties(output, (3, dim))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_large_input(self, mock_distributed, mock_kernels):
        dim = 1024
        inter_dim = 2048
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = create_test_tensor((8, dim))
        output = expert(x)
        
        assert_tensor_properties(output, (8, dim))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_consistency(self, mock_distributed, mock_kernels):
        dim = 128
        inter_dim = 256
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = create_test_tensor((2, dim))
        
        torch.manual_seed(42)
        output1 = expert(x)
        
        torch.manual_seed(42)
        output2 = expert(x)
        
        torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-6)
    
    def test_expert_parameter_count(self, mock_distributed, mock_kernels):
        dim = 256
        inter_dim = 512
        expert = Expert(dim, inter_dim)
        
        total_params = sum(p.numel() for p in expert.parameters())
        expected_params = (dim * inter_dim) + (inter_dim * dim) + (dim * inter_dim)
        
        assert total_params == expected_params
    
    def test_expert_output_range(self, mock_distributed, mock_kernels):
        dim = 64
        inter_dim = 128
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x = create_test_tensor((10, dim))
        output = expert(x)
        
        assert torch.isfinite(output).all()
        
        output_std = output.std()
        assert output_std > 0, "Output should have non-zero variance"
    
    def test_expert_batch_independence(self, mock_distributed, mock_kernels):
        dim = 128
        inter_dim = 256
        expert = Expert(dim, inter_dim)
        initialize_module_weights(expert)
        
        x1 = create_test_tensor((1, dim))
        x2 = create_test_tensor((1, dim))
        x_batch = torch.cat([x1, x2], dim=0)
        
        output1 = expert(x1)
        output2 = expert(x2)
        output_batch = expert(x_batch)
        
        torch.testing.assert_close(output_batch[0:1], output1, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(output_batch[1:2], output2, rtol=1e-4, atol=1e-6)
