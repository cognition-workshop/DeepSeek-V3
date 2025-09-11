import pytest
import torch
from unittest.mock import patch

from inference.model import Block, MoE, MLP, ModelArgs
from tests.test_utils import (
    create_test_tensor, assert_tensor_properties, create_minimal_config,
    initialize_module_weights
)


class TestIntegration:
    
    def test_block_mlp_vs_moe_selection(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_dense_layers=2, n_layers=5)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            dense_block = Block(layer_id=0, args=config)
            moe_block = Block(layer_id=3, args=config)
            
            assert isinstance(dense_block.ffn, MLP)
            assert isinstance(moe_block.ffn, MoE)
    
    def test_block_forward_with_mlp(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_dense_layers=2, dim=256)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=0, args=config)
            
            batch_size, seq_len = 2, 8
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = block(x, start_pos, freqs_cis, mask)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_block_forward_with_moe(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_dense_layers=1, dim=256)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=2, args=config)
            
            batch_size, seq_len = 2, 8
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = block(x, start_pos, freqs_cis, mask)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_block_residual_connections(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=0, args=config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)), \
                 patch.object(block.ffn, 'forward', return_value=torch.zeros_like(x)):
                
                output = block(x, start_pos, freqs_cis, mask)
                
                torch.testing.assert_close(output, x, rtol=1e-5, atol=1e-6)
    
    def test_block_with_attention_mask(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=0, args=config)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = block(x, start_pos, freqs_cis, mask)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
    
    def test_multiple_blocks_sequence(self, mock_distributed, mock_kernels):
        config = create_minimal_config(n_dense_layers=1, n_layers=3, dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            blocks = [Block(layer_id=i, args=config) for i in range(3)]
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            h = x
            for block in blocks:
                with patch.object(block.attn, 'forward', return_value=torch.zeros_like(h)):
                    h = block(h, start_pos, freqs_cis, mask)
            
            assert_tensor_properties(h, (batch_size, seq_len, config.dim))
    
    def test_block_layer_normalization(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=0, args=config)
            
            assert hasattr(block, 'attn_norm')
            assert hasattr(block, 'ffn_norm')
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            
            attn_norm_out = block.attn_norm(x)
            ffn_norm_out = block.ffn_norm(x)
            
            assert_tensor_properties(attn_norm_out, (batch_size, seq_len, config.dim))
            assert_tensor_properties(ffn_norm_out, (batch_size, seq_len, config.dim))
    
    def test_end_to_end_moe_flow(self, mock_distributed, mock_kernels):
        config = create_minimal_config(
            n_dense_layers=1,
            n_routed_experts=8,
            n_activated_experts=2,
            dim=256
        )
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            moe_block = Block(layer_id=2, args=config)
            initialize_module_weights(moe_block)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(moe_block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = moe_block(x, start_pos, freqs_cis, mask)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()
    
    def test_gradient_flow_through_block(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=0, args=config)
            initialize_module_weights(block)
            
            batch_size, seq_len = 2, 4
            x = create_test_tensor((batch_size, seq_len, config.dim))
            x.requires_grad_(True)
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = block(x, start_pos, freqs_cis, mask)
                loss = output.sum()
                loss.backward()
                
                assert x.grad is not None
                assert not torch.isnan(x.grad).any()
    
    def test_block_parameter_sharing(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=128)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block1 = Block(layer_id=0, args=config)
            block2 = Block(layer_id=0, args=config)
            
            assert block1.ffn is not block2.ffn
            assert block1.attn is not block2.attn
            assert block1.attn_norm is not block2.attn_norm
            assert block1.ffn_norm is not block2.ffn_norm
    
    def test_different_layer_configurations(self, mock_distributed, mock_kernels):
        test_configs = [
            {'n_dense_layers': 0, 'n_layers': 3},
            {'n_dense_layers': 1, 'n_layers': 3},
            {'n_dense_layers': 2, 'n_layers': 3},
            {'n_dense_layers': 3, 'n_layers': 3},
        ]
        
        for config_dict in test_configs:
            config = create_minimal_config(**config_dict, dim=128)
            
            with patch('inference.model.world_size', 1), \
                 patch('inference.model.rank', 0):
                
                blocks = [Block(layer_id=i, args=config) for i in range(config.n_layers)]
                
                for i, block in enumerate(blocks):
                    if i < config.n_dense_layers:
                        assert isinstance(block.ffn, MLP)
                    else:
                        assert isinstance(block.ffn, MoE)
    
    def test_block_memory_efficiency(self, mock_distributed, mock_kernels):
        config = create_minimal_config(dim=1024, n_routed_experts=64)
        
        with patch('inference.model.world_size', 1), \
             patch('inference.model.rank', 0):
            
            block = Block(layer_id=2, args=config)
            
            batch_size, seq_len = 8, 16
            x = create_test_tensor((batch_size, seq_len, config.dim))
            start_pos = 0
            freqs_cis = torch.randn(seq_len, config.qk_rope_head_dim // 2, dtype=torch.complex64)
            mask = None
            
            with patch.object(block.attn, 'forward', return_value=torch.zeros_like(x)):
                output = block(x, start_pos, freqs_cis, mask)
                
                assert_tensor_properties(output, (batch_size, seq_len, config.dim))
