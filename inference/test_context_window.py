"""
Test script for the context window testing application.

This script tests the functionality of the context window testing application
without requiring actual model weights.
"""

import unittest
import torch
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.context_window_test import NIAHTest, ContextWindowTester, CONTEXT_LENGTHS


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.eos_token_id = 2
    
    def encode(self, text):
        """Mock encode method that returns a list of token IDs."""
        return [hash(word) % 10000 for word in text.split()]
    
    def decode(self, tokens, skip_special_tokens=True):
        """Mock decode method that returns a string."""
        return f"Decoded: {tokens}"


class MockTransformer:
    """Mock transformer model for testing."""
    
    def __init__(self, args):
        self.args = args


class TestNIAHTest(unittest.TestCase):
    """Test the NIAHTest class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.tokenizer = MockTokenizer()
        self.niah_test = NIAHTest(self.tokenizer)
    
    def test_create_test_data(self):
        """Test the create_test_data method."""
        needle_text = "This is a test needle"
        
        for position in [0.25, 0.5, 0.75]:
            context_length = 100
            tokens, needle, start_pos = self.niah_test.create_test_data(
                context_length=context_length,
                needle_text=needle_text,
                needle_position=position
            )
            
            self.assertIsInstance(tokens, list)
            self.assertIsInstance(needle, str)
            self.assertIsInstance(start_pos, int)
            
            self.assertEqual(needle, needle_text)
            
            self.assertLessEqual(len(tokens), context_length)
            
            expected_pos = int(position * (context_length - len(self.tokenizer.encode(needle_text))))
            self.assertAlmostEqual(start_pos, expected_pos, delta=5)
    
    @patch('inference.context_window_test.generate')
    def test_evaluate_retrieval(self, mock_generate):
        """Test the evaluate_retrieval method."""
        mock_generate.return_value = [[1, 2, 3]]
        
        model = MockTransformer(MagicMock())
        
        context_tokens = [1, 2, 3, 4, 5]
        needle_text = "test needle"
        
        with patch.object(self.tokenizer, 'decode', return_value=f"Response containing {needle_text}"):
            success, response, time = self.niah_test.evaluate_retrieval(
                model=model,
                context_tokens=context_tokens,
                needle_text=needle_text
            )
            
            self.assertIsInstance(success, bool)
            self.assertIsInstance(response, str)
            self.assertIsInstance(time, float)
            
            self.assertTrue(success)
        
        with patch.object(self.tokenizer, 'decode', return_value="Response without the needle"):
            success, response, time = self.niah_test.evaluate_retrieval(
                model=model,
                context_tokens=context_tokens,
                needle_text=needle_text
            )
            
            self.assertFalse(success)


class TestContextWindowTester(unittest.TestCase):
    """Test the ContextWindowTester class."""
    
    @patch('inference.context_window_test.Transformer')
    @patch('inference.context_window_test.AutoTokenizer')
    @patch('inference.context_window_test.generate')
    @patch('inference.context_window_test.load_model')
    def setUp(self, mock_load_model, mock_generate, mock_tokenizer, mock_transformer):
        """Set up the test environment."""
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()
        mock_transformer.return_value = MockTransformer(MagicMock())
        mock_generate.return_value = [[1, 2, 3]]
        
        import json
        import tempfile
        self.temp_config = tempfile.NamedTemporaryFile(delete=False, mode='w')
        json.dump({
            "vocab_size": 10000,
            "dim": 1024,
            "inter_dim": 4096,
            "n_layers": 2,
            "n_heads": 16,
            "max_seq_len": 4096,
            "original_seq_len": 4096,
            "rope_theta": 10000.0,
            "rope_factor": 40,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0
        }, self.temp_config)
        self.temp_config.close()
        
        with patch('builtins.open', return_value=self.temp_config):
            self.tester = ContextWindowTester(
                ckpt_path="/mock/path",
                config_path=self.temp_config.name
            )
    
    def tearDown(self):
        """Clean up after the test."""
        os.unlink(self.temp_config.name)
    
    @patch('inference.context_window_test.generate')
    def test_run_niah_test(self, mock_generate):
        """Test the run_niah_test method."""
        mock_generate.return_value = [[1, 2, 3]]
        
        needle_text = "This is a test needle"
        context_length = 100
        
        with patch.object(self.tester.tokenizer, 'decode', return_value=f"Response containing {needle_text}"):
            result = self.tester.run_niah_test(
                context_length=context_length,
                needle_text=needle_text,
                needle_positions=[0.5],
                repetitions=1
            )
            
            self.assertIsInstance(result, dict)
            
            self.assertIn("context_length", result)
            self.assertIn("needle_text", result)
            self.assertIn("tests", result)
            
            self.assertEqual(result["context_length"], context_length)
            self.assertEqual(result["needle_text"], needle_text)
            
            self.assertEqual(len(result["tests"]), 1)
            test = result["tests"][0]
            self.assertIn("position", test)
            self.assertIn("runs", test)
            self.assertIn("success_rate", test)
            self.assertIn("avg_inference_time", test)
            
            self.assertEqual(test["success_rate"], 1.0)
    
    @patch('inference.context_window_test.generate')
    def test_run_custom_text_test(self, mock_generate):
        """Test the run_custom_text_test method."""
        mock_generate.return_value = [[1, 2, 3]]
        
        text = "This is a test text"
        prompt = "Test prompt"
        
        result = self.tester.run_custom_text_test(
            text=text,
            prompt=prompt
        )
        
        self.assertIsInstance(result, dict)
        
        self.assertIn("context_length", result)
        self.assertIn("response", result)
        self.assertIn("inference_time", result)
        self.assertIn("tokens_per_second", result)
        
        self.assertEqual(result["context_length"], len(self.tester.tokenizer.encode(text)))
    
    def test_memory_usage(self):
        """Test memory usage for long contexts."""
        long_context = "word " * 10000  # Approximately 10K tokens
        
        import psutil
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        tokens = self.tester.tokenizer.encode(long_context)
        
        after_tokenize_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = after_tokenize_memory - initial_memory
        print(f"Memory increase for 10K tokens: {memory_increase:.2f} MB")
        
        self.assertLess(memory_increase, 100)


if __name__ == "__main__":
    unittest.main()
