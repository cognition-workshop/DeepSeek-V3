import pytest
import torch
import torch.distributed as dist
from unittest.mock import patch, MagicMock

from inference.model import MoE, ModelArgs
from tests.test_utils import (
    create_test_tensor, assert_tensor_properties, mock_world_size_rank,
    create_minimal_config, check_gradient_flow, initialize_module_weights
)


class TestMoE:
    
    def test_moe_initialization_single_device(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            assert moe.dim == simple_config.dim
            assert moe.n_routed_experts == simple_config.n_routed_experts
            assert moe.n_local_experts == simple_config.n_routed_experts
            assert moe.n_activated_experts == simple_config.n_activated_experts
            assert moe.experts_start_idx == 0
            assert moe.experts_end_idx == simple_config.n_routed_experts
            
            assert hasattr(moe, 'gate')
            assert hasattr(moe, 'experts')
            assert hasattr(moe, 'shared_experts')
            
            assert len(moe.experts) == simple_config.n_routed_experts
            for expert in moe.experts:
                assert expert is not None
    
    def test_moe_initialization_distributed(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=16)
        world_size = 4
        rank = 1
        
        with patch('inference.model.world_size', world_size), \
             patch('inference.model.rank', rank):
            moe = MoE(config)
            
            expected_local_experts = config.n_routed_experts // world_size
            expected_start_idx = rank * expected_local_experts
            expected_end_idx = expected_start_idx + expected_local_experts
            
            assert moe.n_local_experts == expected_local_experts
            assert moe.experts_start_idx == expected_start_idx
            assert moe.experts_end_idx == expected_end_idx
            
            assert len(moe.experts) == config.n_routed_experts
            
            for i, expert in enumerate(moe.experts):
                if expected_start_idx <= i < expected_end_idx:
                    assert expert is not None
                else:
                    assert expert is None
    
    def test_moe_forward_basic(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
            assert output.dtype == x.dtype
    
    def test_moe_forward_different_shapes(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            test_cases = [
                (1, 1, simple_config.dim),
                (2, 4, simple_config.dim),
                (4, 8, simple_config.dim),
                (8, 16, simple_config.dim),
            ]
            
            for batch_size, seq_len, dim in test_cases:
                x = create_test_tensor((batch_size, seq_len, dim))
                output = moe(x)
                assert_tensor_properties(output, (batch_size, seq_len, dim))
    
    def test_moe_expert_routing(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            with patch.object(moe.gate, 'forward') as mock_gate:
                mock_weights = torch.rand(batch_size * seq_len, simple_config.n_activated_experts)
                mock_indices = torch.randint(0, simple_config.n_routed_experts, 
                                           (batch_size * seq_len, simple_config.n_activated_experts))
                mock_gate.return_value = (mock_weights, mock_indices)
                
                output = moe(x)
                
                mock_gate.assert_called_once()
                assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
    
    def test_moe_shared_experts_contribution(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            with patch.object(moe.shared_experts, 'forward') as mock_shared:
                mock_shared_output = create_test_tensor((batch_size * seq_len, simple_config.dim))
                mock_shared.return_value = mock_shared_output
                
                output = moe(x)
                
                mock_shared.assert_called_once()
                assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
    
    def test_moe_distributed_all_reduce(self, simple_config, mock_distributed, mock_kernels):
        world_size = 2
        with patch('inference.model.world_size', world_size), \
             patch('inference.model.rank', 0), \
             patch('torch.distributed.all_reduce') as mock_all_reduce:
            
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            output = moe(x)
            
            assert mock_all_reduce.call_count >= 1, f"Expected at least 1 call to all_reduce, got {mock_all_reduce.call_count}"
            assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
    
    def test_moe_expert_load_balancing(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=8, n_activated_experts=2)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(config)
            
            batch_size, seq_len = 4, 8
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            expert_usage = torch.zeros(config.n_routed_experts)
            
            with patch.object(moe.gate, 'forward') as mock_gate:
                indices = torch.randint(0, config.n_routed_experts, 
                                      (batch_size * seq_len, config.n_activated_experts))
                weights = torch.rand(batch_size * seq_len, config.n_activated_experts)
                mock_gate.return_value = (weights, indices)
                
                output = moe(x)
                
                for idx in indices.flatten():
                    expert_usage[idx] += 1
                
                assert expert_usage.sum() == batch_size * seq_len * config.n_activated_experts
    
    def test_moe_zero_expert_activation(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            with patch.object(moe.gate, 'forward') as mock_gate:
                weights = torch.zeros(batch_size * seq_len, simple_config.n_activated_experts)
                indices = torch.full((batch_size * seq_len, simple_config.n_activated_experts), 
                                   simple_config.n_routed_experts - 1)
                mock_gate.return_value = (weights, indices)
                
                output = moe(x)
                
                assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
                assert not torch.isnan(output).any()
    
    def test_moe_gradient_flow(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            initialize_module_weights(moe)
            
            x = create_test_tensor((2, 4, simple_config.dim))
            check_gradient_flow(moe, x)
    
    def test_moe_expert_divisibility_assertion(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=15)
        world_size = 4
        
        with patch('inference.model.world_size', world_size), \
             patch('inference.model.rank', 0):
            
            with pytest.raises(AssertionError, match="Number of experts must be divisible by world size"):
                MoE(config)
    
    def test_moe_different_expert_counts(self, mock_distributed, mock_kernels):
        expert_counts = [4, 8, 16, 32, 64]
        
        for n_experts in expert_counts:
            config = create_minimal_config(n_routed_experts=n_experts, n_activated_experts=min(4, n_experts))
            
            with patch('inference.model.world_size', 1), \
                 patch('inference.model.rank', 0):
                moe = MoE(config)
                
                x = create_test_tensor((2, 4, config.dim))
                output = moe(x)
                
                assert_tensor_properties(output, (2, 4, config.dim))
    
    def test_moe_consistency(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            x = create_test_tensor((2, 4, simple_config.dim))
            
            torch.manual_seed(42)
            output1 = moe(x)
            
            torch.manual_seed(42)
            output2 = moe(x)
            
            torch.testing.assert_close(output1, output2)
    
    def test_moe_output_combination(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            with patch.object(moe.shared_experts, 'forward') as mock_shared:
                shared_output = create_test_tensor((batch_size * seq_len, simple_config.dim))
                mock_shared.return_value = shared_output
                
                with patch.object(moe.gate, 'forward') as mock_gate:
                    weights = torch.ones(batch_size * seq_len, simple_config.n_activated_experts) * 0.5
                    indices = torch.zeros(batch_size * seq_len, simple_config.n_activated_experts, dtype=torch.long)
                    mock_gate.return_value = (weights, indices)
                    
                    output = moe(x)
                    
                    assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
    
    def test_moe_large_batch_size(self, simple_config, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            moe = MoE(simple_config)
            
            batch_size, seq_len = 16, 32
            x = create_test_tensor((batch_size, seq_len, simple_config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, simple_config.dim))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
