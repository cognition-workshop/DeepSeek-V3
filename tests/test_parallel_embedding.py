import pytest
import torch
import torch.nn.functional as F
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))

from model import ParallelEmbedding


class TestParallelEmbedding:
    
    @pytest.fixture
    def embedding_params(self):
        return {
            'vocab_size': 1000,
            'dim': 128
        }
    
    @pytest.fixture
    def sample_input(self):
        return torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    
    def test_single_device_initialization(self, embedding_params):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            
            assert embedding.vocab_size == 1000
            assert embedding.dim == 128
            assert embedding.part_vocab_size == 1000
            assert embedding.vocab_start_idx == 0
            assert embedding.vocab_end_idx == 1000
            assert embedding.weight.shape == (1000, 128)
    
    def test_multi_device_initialization(self, embedding_params):
        with patch('model.world_size', 4), patch('model.rank', 2):
            embedding = ParallelEmbedding(**embedding_params)
            
            assert embedding.vocab_size == 1000
            assert embedding.dim == 128
            assert embedding.part_vocab_size == 250
            assert embedding.vocab_start_idx == 500
            assert embedding.vocab_end_idx == 750
            assert embedding.weight.shape == (250, 128)
    
    def test_vocab_size_divisibility_assertion(self):
        with patch('model.world_size', 3), patch('model.rank', 0):
            with pytest.raises(AssertionError, match="Vocabulary size must be divisible by world size"):
                ParallelEmbedding(vocab_size=1000, dim=128)
    
    @pytest.mark.parametrize("world_size,rank,expected_start,expected_end", [
        (1, 0, 0, 1000),
        (2, 0, 0, 500),
        (2, 1, 500, 1000),
        (4, 0, 0, 250),
        (4, 1, 250, 500),
        (4, 2, 500, 750),
        (4, 3, 750, 1000),
    ])
    def test_vocabulary_partitioning(self, world_size, rank, expected_start, expected_end):
        with patch('model.world_size', world_size), patch('model.rank', rank):
            embedding = ParallelEmbedding(vocab_size=1000, dim=128)
            
            assert embedding.vocab_start_idx == expected_start
            assert embedding.vocab_end_idx == expected_end
            assert embedding.part_vocab_size == (expected_end - expected_start)
    
    def test_single_device_forward(self, embedding_params, sample_input):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            output = embedding(sample_input)
            
            assert output.shape == (2, 4, 128)
            assert output.dtype == embedding.weight.dtype
    
    @patch('model.dist')
    def test_multi_device_forward_with_masking(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            input_tokens = torch.tensor([[100, 300, 600, 800]])
            output = embedding(input_tokens)
            
            assert output.shape == (1, 4, 128)
            mock_dist.all_reduce.assert_called_once_with(output)
    
    @patch('model.dist')
    def test_masking_behavior(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.ones_(embedding.weight)
            
            input_tokens = torch.tensor([[100, 300, 600, 800]])
            output = embedding(input_tokens)
            
            expected_mask = torch.tensor([[True, False, True, True]])
            
            assert torch.allclose(output[expected_mask], torch.zeros_like(output[expected_mask]))
            mock_dist.all_reduce.assert_called_once()
    
    def test_vocabulary_boundary_tokens(self, embedding_params):
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            boundary_tokens = torch.tensor([[249, 250, 499, 500]])
            
            with patch('model.dist') as mock_dist:
                mock_dist.all_reduce = MagicMock()
                output = embedding(boundary_tokens)
                
                assert output.shape == (1, 4, 128)
                mock_dist.all_reduce.assert_called_once()
    
    def test_empty_input(self, embedding_params):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            
            empty_input = torch.empty((0, 0), dtype=torch.long)
            output = embedding(empty_input)
            
            assert output.shape == (0, 0, 128)
    
    def test_out_of_vocabulary_tokens_single_device(self, embedding_params):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            oov_tokens = torch.tensor([[1000, 1001, 2000]])
            
            with pytest.raises(IndexError):
                embedding(oov_tokens)
    
    @patch('model.dist')
    def test_out_of_vocabulary_tokens_multi_device(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            oov_tokens = torch.tensor([[1000, 1001, 2000]])
            output = embedding(oov_tokens)
            
            assert torch.allclose(output, torch.zeros_like(output))
            mock_dist.all_reduce.assert_called_once()
    
    def test_negative_token_indices(self, embedding_params):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            negative_tokens = torch.tensor([[-1, -5, -10]])
            
            with pytest.raises(IndexError):
                embedding(negative_tokens)
    
    @patch('model.dist')
    def test_negative_token_indices_multi_device(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            negative_tokens = torch.tensor([[-1, -5, -10]])
            output = embedding(negative_tokens)
            
            assert torch.allclose(output, torch.zeros_like(output))
            mock_dist.all_reduce.assert_called_once()
    
    @patch('model.dist')
    def test_distributed_aggregation_not_called_single_device(self, mock_dist, embedding_params, sample_input):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            output = embedding(sample_input)
            
            mock_dist.all_reduce.assert_not_called()
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),
        (1, 10),
        (4, 8),
        (16, 32),
    ])
    def test_various_input_shapes(self, embedding_params, batch_size, seq_len):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
            
            input_tokens = torch.randint(0, 1000, (batch_size, seq_len))
            output = embedding(input_tokens)
            
            assert output.shape == (batch_size, seq_len, 128)
    
    @pytest.mark.parametrize("vocab_size,dim", [
        (512, 64),
        (2048, 256),
        (4096, 512),
        (8192, 1024),
    ])
    def test_various_embedding_dimensions(self, vocab_size, dim):
        with patch('model.world_size', 1), patch('model.rank', 0):
            embedding = ParallelEmbedding(vocab_size=vocab_size, dim=dim)
            
            assert embedding.vocab_size == vocab_size
            assert embedding.dim == dim
            assert embedding.weight.shape == (vocab_size, dim)
    
    @patch('model.dist')
    def test_token_offset_calculation(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 2):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.constant_(embedding.weight, 1.0)
            
            input_tokens = torch.tensor([[500, 600, 700]])
            
            original_input = input_tokens.clone()
            output = embedding(input_tokens)
            
            assert torch.equal(input_tokens, original_input)
            mock_dist.all_reduce.assert_called_once()
    
    @patch('model.dist')
    def test_mask_preservation_through_forward_pass(self, mock_dist, embedding_params):
        mock_dist.all_reduce = MagicMock()
        
        with patch('model.world_size', 4), patch('model.rank', 1):
            embedding = ParallelEmbedding(**embedding_params)
            torch.nn.init.ones_(embedding.weight)
            
            input_tokens = torch.tensor([[100, 300, 600]])
            output = embedding(input_tokens)
            
            expected_mask = (input_tokens < 250) | (input_tokens >= 500)
            masked_output = output[expected_mask]
            
            assert torch.allclose(masked_output, torch.zeros_like(masked_output))
            mock_dist.all_reduce.assert_called_once()
