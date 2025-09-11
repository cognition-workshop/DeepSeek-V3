import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch

from inference.model import Gate, ModelArgs
from tests.test_utils import (
    create_test_tensor, assert_tensor_properties, mock_world_size_rank,
    create_minimal_config, verify_routing_properties, initialize_gate_weights
)


class TestGate:
    
    def test_gate_initialization_basic(self, simple_config, mock_distributed, mock_kernels):
        gate = Gate(simple_config)
        
        assert gate.dim == simple_config.dim
        assert gate.topk == simple_config.n_activated_experts
        assert gate.n_groups == simple_config.n_expert_groups
        assert gate.topk_groups == simple_config.n_limited_groups
        assert gate.score_func == simple_config.score_func
        assert gate.route_scale == simple_config.route_scale
        
        assert gate.weight.shape == (simple_config.n_routed_experts, simple_config.dim)
        assert gate.bias is None
    
    def test_gate_initialization_with_bias(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=7168, n_routed_experts=16)
        gate = Gate(config)
        
        assert gate.bias is not None
        assert gate.bias.shape == (config.n_routed_experts,)
    
    def test_gate_initialization_without_bias(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=512, n_routed_experts=16)
        gate = Gate(config)
        
        assert gate.bias is None
    
    def test_gate_forward_softmax(self, simple_config, mock_distributed, mock_kernels):
        config = create_minimal_config(score_func="softmax")
        gate = Gate(config)
        initialize_gate_weights(gate)
        
        batch_size, seq_len = 2, 4
        x = create_test_tensor((batch_size * seq_len, config.dim))
        
        weights, indices = gate(x)
        
        verify_routing_properties(weights, indices, config.n_activated_experts, config.n_routed_experts)
        
        for i in range(batch_size * seq_len):
            assert weights[i].sum() > 0, "Weight sum should be positive"
    
    def test_gate_forward_sigmoid(self, mock_distributed, mock_kernels):
        config = create_minimal_config(score_func="sigmoid", route_scale=2.0)
        gate = Gate(config)
        initialize_gate_weights(gate)
        
        batch_size, seq_len = 2, 4
        x = create_test_tensor((batch_size * seq_len, config.dim))
        
        weights, indices = gate(x)
        
        verify_routing_properties(weights, indices, config.n_activated_experts, config.n_routed_experts)
        
        for i in range(batch_size * seq_len):
            torch.testing.assert_close(weights[i].sum(), torch.tensor(config.route_scale), rtol=1e-2, atol=1e-2)
    
    def test_gate_expert_groups_routing(self, mock_distributed, mock_kernels):
        config = create_minimal_config(
            n_routed_experts=16,
            n_activated_experts=4,
            n_expert_groups=4,
            n_limited_groups=2
        )
        gate = Gate(config)
        
        batch_size = 3
        x = create_test_tensor((batch_size, config.dim))
        
        weights, indices = gate(x)
        
        verify_routing_properties(weights, indices, config.n_activated_experts, config.n_routed_experts)
    
    def test_gate_with_bias(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=7168, n_routed_experts=8)
        gate = Gate(config)
        initialize_gate_weights(gate)
        
        batch_size = 2
        x = create_test_tensor((batch_size, config.dim))
        
        weights, indices = gate(x)
        
        verify_routing_properties(weights, indices, config.n_activated_experts, config.n_routed_experts)
    
    def test_gate_different_topk_values(self, mock_distributed, mock_kernels):
        for topk in [1, 2, 4, 6]:
            config = create_minimal_config(n_activated_experts=topk, n_routed_experts=16)
            gate = Gate(config)
            
            batch_size = 2
            x = create_test_tensor((batch_size, config.dim))
            
            weights, indices = gate(x)
            
            verify_routing_properties(weights, indices, topk, config.n_routed_experts)
    
    def test_gate_output_shapes(self, simple_config, mock_distributed, mock_kernels):
        gate = Gate(simple_config)
        
        test_cases = [
            (1, simple_config.dim),
            (4, simple_config.dim),
            (8, simple_config.dim),
        ]
        
        for batch_size, dim in test_cases:
            x = create_test_tensor((batch_size, dim))
            weights, indices = gate(x)
            
            assert_tensor_properties(weights, (batch_size, simple_config.n_activated_experts))
            assert_tensor_properties(indices, (batch_size, simple_config.n_activated_experts))
            assert indices.dtype == torch.long
    
    def test_gate_deterministic_output(self, simple_config, mock_distributed, mock_kernels):
        gate = Gate(simple_config)
        
        x = create_test_tensor((2, simple_config.dim))
        
        torch.manual_seed(42)
        weights1, indices1 = gate(x)
        
        torch.manual_seed(42)
        weights2, indices2 = gate(x)
        
        torch.testing.assert_close(weights1, weights2)
        torch.testing.assert_close(indices1, indices2)
    
    def test_gate_gradient_flow(self, simple_config, mock_distributed, mock_kernels):
        gate = Gate(simple_config)
        initialize_gate_weights(gate)
        
        x = create_test_tensor((2, simple_config.dim))
        x.requires_grad_(True)
        
        weights, indices = gate(x)
        loss = weights.sum()
        loss.backward()
        
        assert gate.weight.grad is not None
        assert not torch.isnan(gate.weight.grad).any()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_gate_weight_normalization(self, mock_distributed, mock_kernels):
        config = create_minimal_config(score_func="sigmoid")
        gate = Gate(config)
        initialize_gate_weights(gate)
        
        x = create_test_tensor((3, config.dim))
        weights, indices = gate(x)
        
        for i in range(x.shape[0]):
            weight_sum = weights[i].sum()
            torch.testing.assert_close(weight_sum, torch.tensor(config.route_scale), rtol=1e-2, atol=1e-2)
    
    def test_gate_large_expert_count(self, mock_distributed, mock_kernels):
        config = create_minimal_config(
            n_routed_experts=256,
            n_activated_experts=8,
            dim=1024
        )
        gate = Gate(config)
        
        x = create_test_tensor((4, config.dim))
        weights, indices = gate(x)
        
        verify_routing_properties(weights, indices, config.n_activated_experts, config.n_routed_experts)
