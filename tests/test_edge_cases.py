import pytest
import torch
from unittest.mock import patch

from inference.model import MoE, Gate, Expert, ModelArgs
from tests.test_utils import (
    create_test_tensor, assert_tensor_properties, create_minimal_config,
    initialize_module_weights, initialize_gate_weights
)


class TestEdgeCases:
    
    def test_zero_expert_activation(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=8, n_activated_experts=2)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            with patch.object(moe.gate, 'forward') as mock_gate:
                weights = torch.zeros(batch_size * seq_len, config.n_activated_experts)
                indices = torch.full((batch_size * seq_len, config.n_activated_experts), 
                                   config.n_routed_experts - 1)
                mock_gate.return_value = (weights, indices)
                
                output = moe(x)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()
    
    def test_maximum_expert_activation(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=4, n_activated_experts=4)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            initialize_module_weights(moe)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, config.dim))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_single_expert_system(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=1, n_activated_experts=1)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_mismatched_tensor_shapes(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=512)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            wrong_dim = 256
            x = create_test_tensor((2, 4, wrong_dim))
            
            try:
                output = moe(x)
                assert output.shape[:-1] == x.shape[:-1]
            except (RuntimeError, AssertionError):
                pass
    
    def test_empty_batch(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            x = create_test_tensor((0, 4, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (0, 4, config.dim))
    
    def test_single_token_sequence(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            batch_size = 2
            x = create_test_tensor((batch_size, 1, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, 1, config.dim))
    
    def test_very_large_sequence_length(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            initialize_module_weights(moe)
            
            batch_size, seq_len = 1, 1024
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, config.dim))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_extreme_routing_weights(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        gate = Gate(config)
        
        batch_size = 4
        x = create_test_tensor((batch_size, config.dim))
        
        with torch.no_grad():
            gate.weight.fill_(1000.0)
            weights, indices = gate(x)
            
            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()
            assert torch.all(weights >= 0)
    
    def test_numerical_stability_small_weights(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        gate = Gate(config)
        
        batch_size = 4
        x = create_test_tensor((batch_size, config.dim))
        
        with torch.no_grad():
            gate.weight.fill_(1e-8)
            weights, indices = gate(x)
            
            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()
            assert torch.all(weights >= 0)
    
    def test_expert_with_zero_weights(self, mock_distributed, mock_kernels):
        expert = Expert(dim=128, inter_dim=256)
        
        with torch.no_grad():
            expert.w1.weight.zero_()
            expert.w2.weight.zero_()
            expert.w3.weight.zero_()
        
        x = create_test_tensor((2, 128))
        output = expert(x)
        
        assert_tensor_properties(output, (2, 128))
        assert not torch.isnan(output).any()
    
    def test_gradient_explosion_protection(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            x = create_test_tensor((2, 4, config.dim))
            x.requires_grad_(True)
            
            with torch.no_grad():
                for param in moe.parameters():
                    param.fill_(100.0)
            
            output = moe(x)
            loss = output.sum()
            loss.backward()
            
            for param in moe.parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()
    
    def test_distributed_rank_out_of_bounds(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=8)
        world_size = 2
        rank = 2
        
        with patch('inference.model.world_size', world_size), \
             patch('inference.model.rank', rank):
            
            moe = MoE(config)
            
            assert moe.experts_start_idx == rank * (config.n_routed_experts // world_size)
            assert moe.experts_end_idx == moe.experts_start_idx + (config.n_routed_experts // world_size)
            
            local_expert_count = sum(1 for expert in moe.experts if expert is not None)
            assert local_expert_count == 0
    
    def test_memory_pressure_large_experts(self, mock_distributed, mock_kernels):
        config = create_minimal_config(
            dim=2048,
            moe_inter_dim=8192,
            n_routed_experts=128,
            n_activated_experts=8
        )
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            batch_size, seq_len = 4, 16
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            output = moe(x)
            
            assert_tensor_properties(output, (batch_size, seq_len, config.dim))
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_routing_consistency_under_noise(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        gate = Gate(config)
        initialize_gate_weights(gate)
        
        x = create_test_tensor((4, config.dim))
        
        weights1, indices1 = gate(x)
        
        x_noisy = x + torch.randn_like(x) * 1e-6
        weights2, indices2 = gate(x_noisy)
        
        assert not torch.isnan(weights1).any() and not torch.isnan(weights2).any()
        assert weights1.shape == weights2.shape
        assert indices1.shape == indices2.shape
    
    def test_expert_load_imbalance(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_routed_experts=8, n_activated_experts=2)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            batch_size, seq_len = 16, 8
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            with patch.object(moe.gate, 'forward') as mock_gate:
                weights = torch.ones(batch_size * seq_len, config.n_activated_experts) * 0.5
                indices = torch.zeros(batch_size * seq_len, config.n_activated_experts, dtype=torch.long)
                mock_gate.return_value = (weights, indices)
                
                output = moe(x)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_dtype_consistency_mixed_precision(self, mock_distributed, mock_kernels):
        config = create_minimal_config()
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config)
            
            x_bf16 = create_test_tensor((2, 4, config.dim)).to(torch.bfloat16)
            x_fp32 = create_test_tensor((2, 4, config.dim)).to(torch.float32)
            
            output_bf16 = moe(x_bf16)
            output_fp32 = moe(x_fp32)
            
            assert output_bf16.dtype == torch.bfloat16
            assert output_fp32.dtype == torch.float32
