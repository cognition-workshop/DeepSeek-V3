import pytest
import torch
import json
from unittest.mock import patch

from inference.model import MoE, Gate, Expert, ModelArgs
from tests.test_utils import create_test_tensor, assert_tensor_properties, initialize_gate_weights


class TestConfigurations:
    
    def test_16b_configuration(self, config_16b, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config_16b)
            
            assert moe.dim == 2048
            assert moe.n_routed_experts == 64
            assert moe.n_activated_experts == 6
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config_16b.dim))
            
            output = moe(x)
            assert_tensor_properties(output, (batch_size, seq_len, config_16b.dim))
    
    def test_236b_configuration(self, config_236b, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config_236b)
            
            assert moe.dim == 5120
            assert moe.n_routed_experts == 160
            assert moe.n_activated_experts == 6
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config_236b.dim))
            
            output = moe(x)
            assert_tensor_properties(output, (batch_size, seq_len, config_236b.dim))
    
    def test_671b_configuration(self, config_671b, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config_671b)
            
            assert moe.dim == 7168
            assert moe.n_routed_experts == 256
            assert moe.n_activated_experts == 8
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config_671b.dim))
            
            output = moe(x)
            assert_tensor_properties(output, (batch_size, seq_len, config_671b.dim))
    
    def test_gate_bias_in_671b(self, config_671b, mock_distributed, mock_kernels):
        gate = Gate(config_671b)
        
        assert gate.bias is not None
        assert gate.bias.shape == (config_671b.n_routed_experts,)
    
    def test_gate_no_bias_in_16b(self, config_16b, mock_distributed, mock_kernels):
        gate = Gate(config_16b)
        
        assert gate.bias is None
    
    def test_expert_groups_236b(self, config_236b, mock_distributed, mock_kernels):
        gate = Gate(config_236b)
        
        assert gate.n_groups == 8
        assert gate.topk_groups == 3
        
        batch_size = 4
        x = create_test_tensor((batch_size, config_236b.dim))
        
        weights, indices = gate(x)
        
        assert_tensor_properties(weights, (batch_size, config_236b.n_activated_experts))
        assert_tensor_properties(indices, (batch_size, config_236b.n_activated_experts))
    
    def test_expert_groups_671b(self, config_671b, mock_distributed, mock_kernels):
        gate = Gate(config_671b)
        
        assert gate.n_groups == 8
        assert gate.topk_groups == 4
        assert gate.score_func == "sigmoid"
        
        batch_size = 4
        x = create_test_tensor((batch_size, config_671b.dim))
        
        weights, indices = gate(x)
        
        assert_tensor_properties(weights, (batch_size, config_671b.n_activated_experts))
        assert_tensor_properties(indices, (batch_size, config_671b.n_activated_experts))
    
    def test_route_scale_differences(self, mock_distributed, mock_kernels):
        configs = [
            (ModelArgs(dim=512, n_routed_experts=8, n_activated_experts=2, route_scale=1.0), 1.0),
            (ModelArgs(dim=512, n_routed_experts=8, n_activated_experts=2, route_scale=2.5), 2.5),
            (ModelArgs(dim=512, n_routed_experts=8, n_activated_experts=2, route_scale=16.0), 16.0),
        ]
        
        for config, expected_scale in configs:
            gate = Gate(config)
            
            with torch.no_grad():
                torch.nn.init.normal_(gate.weight, mean=0.0, std=0.1)
                if gate.bias is not None:
                    torch.nn.init.zeros_(gate.bias)
            
            batch_size = 2
            x = create_test_tensor((batch_size, config.dim))
            
            weights, indices = gate(x)
            
            assert torch.isfinite(weights).all(), "Gate weights contain NaN or Inf values"
            
            for i in range(batch_size):
                weight_sum = weights[i].sum()
                if config.score_func == "sigmoid":
                    torch.testing.assert_close(weight_sum, torch.tensor(expected_scale), rtol=1e-4, atol=1e-6)
                else:
                    assert weight_sum > 0, "Weights should be positive"
                    assert weight_sum <= expected_scale * 1.1, f"Weight sum {weight_sum} should be reasonable for scale {expected_scale}"
    
    def test_shared_experts_scaling(self, mock_distributed, mock_kernels):
        configs = [
            (ModelArgs(dim=512, moe_inter_dim=256, n_shared_experts=1), 256),
            (ModelArgs(dim=512, moe_inter_dim=256, n_shared_experts=2), 512),
            (ModelArgs(dim=512, moe_inter_dim=256, n_shared_experts=4), 1024),
        ]
        
        for config, expected_inter_dim in configs:
            with patch('inference.model.world_size', 1), \
                 patch('inference.model.rank', 0):
                
                moe = MoE(config)
                
                assert moe.shared_experts.w1.out_features == expected_inter_dim
                assert moe.shared_experts.w2.in_features == expected_inter_dim
                assert moe.shared_experts.w3.out_features == expected_inter_dim
    
    def test_distributed_expert_placement(self, config_671b, mock_distributed, mock_kernels):
        world_sizes = [1, 2, 4, 8]
        
        for world_size in world_sizes:
            if config_671b.n_routed_experts % world_size != 0:
                continue
                
            for rank in range(world_size):
                with patch('inference.model.world_size', world_size), \
                     patch('inference.model.rank', rank):
                    
                    moe = MoE(config_671b)
                    
                    expected_local_experts = config_671b.n_routed_experts // world_size
                    expected_start_idx = rank * expected_local_experts
                    expected_end_idx = expected_start_idx + expected_local_experts
                    
                    assert moe.n_local_experts == expected_local_experts
                    assert moe.experts_start_idx == expected_start_idx
                    assert moe.experts_end_idx == expected_end_idx
                    
                    local_expert_count = sum(1 for expert in moe.experts if expert is not None)
                    assert local_expert_count == expected_local_experts
    
    def test_precision_mode_compatibility(self, mock_distributed, mock_kernels):
        base_config = ModelArgs(
            dim=512,
            moe_inter_dim=256,
            n_routed_experts=8,
            n_activated_experts=2
        )
        
        for dtype in ["bf16", "fp8"]:
            config = ModelArgs(**{**base_config.__dict__, 'dtype': dtype})
            
            with patch('inference.model.world_size', 1), \
                 patch('inference.model.rank', 0):
                
                moe = MoE(config)
                
                batch_size, seq_len = 2, 4
                x = create_test_tensor((batch_size, seq_len, config.dim))
                
                output = moe(x)
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_expert_dimension_scaling(self, mock_distributed, mock_kernels):
        test_cases = [
            (2048, 1408),
            (5120, 1536),
            (7168, 2048),
        ]
        
        for dim, moe_inter_dim in test_cases:
            config = ModelArgs(
                dim=dim,
                moe_inter_dim=moe_inter_dim,
                n_routed_experts=8,
                n_activated_experts=2
            )
            
            expert = Expert(config.dim, config.moe_inter_dim)
            
            assert expert.w1.in_features == dim
            assert expert.w1.out_features == moe_inter_dim
            assert expert.w2.in_features == moe_inter_dim
            assert expert.w2.out_features == dim
            assert expert.w3.in_features == dim
            assert expert.w3.out_features == moe_inter_dim
    
    def test_configuration_parameter_validation(self, mock_distributed, mock_kernels):
        valid_configs = [
            {'n_routed_experts': 64, 'n_activated_experts': 6},
            {'n_routed_experts': 160, 'n_activated_experts': 6},
            {'n_routed_experts': 256, 'n_activated_experts': 8},
        ]
        
        for config_dict in valid_configs:
            config = ModelArgs(
                dim=512,
                moe_inter_dim=256,
                **config_dict
            )
            
            with patch('inference.model.world_size', 1), \
                 patch('inference.model.rank', 0):
                
                moe = MoE(config)
                gate = Gate(config)
                
                assert moe.n_routed_experts == config_dict['n_routed_experts']
                assert moe.n_activated_experts == config_dict['n_activated_experts']
                assert gate.topk == config_dict['n_activated_experts']
    
    def test_large_model_memory_efficiency(self, config_671b, mock_distributed, mock_kernels):
        with patch('inference.model.world_size', 4), \
             patch('inference.model.rank', 0):
            
            moe = MoE(config_671b)
            
            local_expert_count = sum(1 for expert in moe.experts if expert is not None)
            expected_local_experts = config_671b.n_routed_experts // 4
            
            assert local_expert_count == expected_local_experts
            
            batch_size, seq_len = 1, 8
            x = create_test_tensor((batch_size, seq_len, config_671b.dim))
            
            output = moe(x)
            assert_tensor_properties(output, (batch_size, seq_len, config_671b.dim))
