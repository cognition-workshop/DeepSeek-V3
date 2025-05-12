"""
Context Window Testing Application for DeepSeek-V3

This module provides utilities to test DeepSeek-V3's performance across different context lengths,
including implementation of the "Needle In A Haystack" (NIAH) test.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional, Union
import random
import numpy as np

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from model import Transformer, ModelArgs
from generate import generate

CONTEXT_LENGTHS = {
    "4k": 4 * 1024,
    "16k": 16 * 1024,
    "32k": 32 * 1024,
    "64k": 64 * 1024,
    "128k": 128 * 1024,
}

class NIAHTest:
    """
    Implementation of the "Needle In A Haystack" (NIAH) test for evaluating model's
    ability to retrieve information from long contexts.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize the NIAH test with a tokenizer.
        
        Args:
            tokenizer: The tokenizer for encoding/decoding text.
        """
        self.tokenizer = tokenizer
        
    def create_test_data(
        self, 
        context_length: int, 
        needle_text: str, 
        needle_position: float = 0.5
    ) -> Tuple[List[int], str, int]:
        """
        Create test data for the NIAH test by placing a "needle" within a "haystack".
        
        Args:
            context_length: The desired context length in tokens.
            needle_text: The text to use as the needle.
            needle_position: The relative position (0.0 to 1.0) to place the needle.
                             Default is 0.5 (middle of the context).
                             
        Returns:
            Tuple containing:
            - List of token IDs for the full context
            - The needle text
            - The token position where the needle starts
        """
        needle_tokens = self.tokenizer.encode(needle_text)
        needle_token_count = len(needle_tokens)
        
        remaining_tokens = context_length - needle_token_count
        if remaining_tokens <= 0:
            raise ValueError(f"Needle text is too long for context length {context_length}")
        
        insert_position = int(remaining_tokens * needle_position)
        
        filler_text_before = "A " * insert_position
        filler_text_after = "B " * (remaining_tokens - insert_position)
        
        filler_tokens_before = self.tokenizer.encode(filler_text_before)
        filler_tokens_after = self.tokenizer.encode(filler_text_after)
        
        total_length = len(filler_tokens_before) + needle_token_count + len(filler_tokens_after)
        if total_length > context_length:
            trim_amount = total_length - context_length
            filler_tokens_after = filler_tokens_after[:-trim_amount]
        elif total_length < context_length:
            padding_needed = context_length - total_length
            filler_tokens_after.extend([self.tokenizer.encode(" C")[0]] * padding_needed)
        
        all_tokens = filler_tokens_before + needle_tokens + filler_tokens_after
        needle_start_position = len(filler_tokens_before)
        
        return all_tokens, needle_text, needle_start_position
    
    def evaluate_retrieval(
        self, 
        model: Transformer, 
        context_tokens: List[int], 
        needle_text: str, 
        max_new_tokens: int = 50, 
        temperature: float = 0.7
    ) -> Tuple[bool, str, float]:
        """
        Evaluate if the model can retrieve the needle text from the context.
        
        Args:
            model: The DeepSeek-V3 model.
            context_tokens: The context tokens including the needle.
            needle_text: The needle text that should be retrieved.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            
        Returns:
            Tuple containing:
            - Boolean indicating whether the needle was retrieved
            - The model's response
            - The time taken for inference
        """
        prompt = f"In the text I just shared with you, there is a specific piece of information. " \
                 f"What is that specific piece of information? Extract and tell me exactly what it is."
        
        prompt_tokens = self.tokenizer.encode(prompt)
        
        full_input = context_tokens + prompt_tokens
        
        start_time = time.time()
        
        completion_tokens = generate(
            model, 
            [full_input], 
            max_new_tokens=max_new_tokens, 
            eos_id=self.tokenizer.eos_token_id, 
            temperature=temperature
        )
        
        inference_time = time.time() - start_time
        
        completion = self.tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        
        needle_retrieved = needle_text.lower() in completion.lower()
        
        return needle_retrieved, completion, inference_time


class ContextWindowTester:
    """
    Main class for testing DeepSeek-V3's performance across different context lengths.
    """
    
    def __init__(
        self, 
        ckpt_path: str, 
        config_path: str,
        device: str = "cuda"
    ):
        """
        Initialize the tester with model paths and configuration.
        
        Args:
            ckpt_path: Path to the model checkpoint.
            config_path: Path to the model configuration file.
            device: Device to run the model on.
        """
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.device = device
        
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        if self.world_size > 1:
            dist.init_process_group("nccl")
        
        torch.cuda.set_device(self.local_rank)
        torch.set_default_dtype(torch.bfloat16)
        
        with open(config_path) as f:
            self.args = ModelArgs(**json.load(f))
            
        with torch.device(device):
            self.model = Transformer(self.args)
            
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        
        self.niah_test = NIAHTest(self.tokenizer)
        
        self._load_model()
    
    def _load_model(self):
        """Load the model weights."""
        from safetensors.torch import load_model
        
        self.tokenizer.decode(generate(
            self.model, 
            [self.tokenizer.encode("DeepSeek")], 
            2, 
            -1, 
            1.0
        )[0])
        
        load_model(
            self.model, 
            os.path.join(self.ckpt_path, f"model{self.rank}-mp{self.world_size}.safetensors")
        )
        
        print(f"Model loaded successfully from {self.ckpt_path}")
    
    def run_niah_test(
        self, 
        context_length: int, 
        needle_text: str = "The capital of France is Paris, and it is known for the Eiffel Tower.",
        needle_positions: List[float] = [0.25, 0.5, 0.75],
        repetitions: int = 3,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> Dict:
        """
        Run the NIAH test with the specified parameters.
        
        Args:
            context_length: The context length to test.
            needle_text: The text to use as the needle.
            needle_positions: List of relative positions to place the needle.
            repetitions: Number of repetitions for each position.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            
        Returns:
            Dictionary containing test results.
        """
        results = {
            "context_length": context_length,
            "needle_text": needle_text,
            "tests": []
        }
        
        for position in needle_positions:
            position_results = {
                "position": position,
                "runs": []
            }
            
            success_count = 0
            total_time = 0.0
            
            for _ in range(repetitions):
                try:
                    context_tokens, _, _ = self.niah_test.create_test_data(
                        context_length=context_length,
                        needle_text=needle_text,
                        needle_position=position
                    )
                    
                    success, response, inference_time = self.niah_test.evaluate_retrieval(
                        model=self.model,
                        context_tokens=context_tokens,
                        needle_text=needle_text,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature
                    )
                    
                    success_count += 1 if success else 0
                    total_time += inference_time
                    
                    position_results["runs"].append({
                        "success": success,
                        "response": response,
                        "inference_time": inference_time
                    })
                    
                except Exception as e:
                    position_results["runs"].append({
                        "error": str(e)
                    })
            
            position_results["success_rate"] = success_count / repetitions if repetitions > 0 else 0
            position_results["avg_inference_time"] = total_time / repetitions if repetitions > 0 else 0
            
            results["tests"].append(position_results)
        
        return results
    
    def run_custom_text_test(
        self,
        text: str,
        prompt: str = "Summarize the above information concisely.",
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> Dict:
        """
        Run a test with custom text input.
        
        Args:
            text: The text to use as context.
            prompt: The prompt to append to the context.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for sampling.
            
        Returns:
            Dictionary containing test results.
        """
        text_tokens = self.tokenizer.encode(text)
        prompt_tokens = self.tokenizer.encode(prompt)
        
        context_length = len(text_tokens)
        
        full_input = text_tokens + prompt_tokens
        
        if len(full_input) > self.args.max_seq_len:
            return {
                "error": f"Input too long ({len(full_input)} tokens) for model's maximum sequence length ({self.args.max_seq_len})"
            }
        
        start_time = time.time()
        
        completion_tokens = generate(
            self.model, 
            [full_input], 
            max_new_tokens=max_new_tokens, 
            eos_id=self.tokenizer.eos_token_id, 
            temperature=temperature
        )
        
        inference_time = time.time() - start_time
        
        completion = self.tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        
        return {
            "context_length": context_length,
            "response": completion,
            "inference_time": inference_time,
            "tokens_per_second": max_new_tokens / inference_time if inference_time > 0 else 0
        }
    
    def load_text_from_file(self, file_path: str) -> str:
        """
        Load text from a file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            The loaded text.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def run_context_length_benchmark(
        self,
        context_lengths: List[int] = None,
        needle_text: str = "The capital of France is Paris, and it is known for the Eiffel Tower.",
        repetitions: int = 3
    ) -> Dict:
        """
        Run a benchmark across different context lengths.
        
        Args:
            context_lengths: List of context lengths to test.
            needle_text: The text to use as the needle.
            repetitions: Number of repetitions for each test.
            
        Returns:
            Dictionary containing benchmark results.
        """
        if context_lengths is None:
            context_lengths = list(CONTEXT_LENGTHS.values())
        
        results = {
            "context_lengths": context_lengths,
            "needle_text": needle_text,
            "results": []
        }
        
        for length in context_lengths:
            try:
                print(f"Testing context length: {length}")
                result = self.run_niah_test(
                    context_length=length,
                    needle_text=needle_text,
                    repetitions=repetitions
                )
                results["results"].append(result)
            except Exception as e:
                results["results"].append({
                    "context_length": length,
                    "error": str(e)
                })
        
        return results


def main():
    """
    Main function to run the context window testing application.
    """
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Context Window Testing")
    
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration file")
    
    parser.add_argument("--test-type", type=str, choices=["niah", "custom"], default="niah", 
                        help="Type of test to run (NIAH or custom text)")
    
    parser.add_argument("--context-length", type=str, choices=list(CONTEXT_LENGTHS.keys()), default="16k",
                        help="Context length for NIAH test")
    parser.add_argument("--needle-text", type=str, 
                        default="The capital of France is Paris, and it is known for the Eiffel Tower.",
                        help="Text to use as the needle in NIAH test")
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark across all context lengths")
    
    parser.add_argument("--input-file", type=str, help="Path to input text file")
    parser.add_argument("--input-text", type=str, help="Custom input text")
    parser.add_argument("--prompt", type=str, default="Summarize the above information concisely.",
                        help="Prompt to append to the input text")
    
    parser.add_argument("--max-new-tokens", type=int, default=100, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for sampling")
    
    args = parser.parse_args()
    
    tester = ContextWindowTester(
        ckpt_path=args.ckpt_path,
        config_path=args.config,
    )
    
    if args.test_type == "niah":
        if args.benchmark:
            results = tester.run_context_length_benchmark()
            
            print("\n=== NIAH Benchmark Results ===")
            for result in results["results"]:
                if "error" in result:
                    print(f"Context length {result['context_length']}: Error - {result['error']}")
                    continue
                
                print(f"\nContext length: {result['context_length']}")
                
                for test in result["tests"]:
                    position = test["position"]
                    success_rate = test["success_rate"]
                    avg_time = test["avg_inference_time"]
                    
                    print(f"  Position {position:.2f}: Success rate: {success_rate:.2f}, Avg time: {avg_time:.2f}s")
            
        else:
            context_length = CONTEXT_LENGTHS[args.context_length]
            result = tester.run_niah_test(
                context_length=context_length,
                needle_text=args.needle_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            print("\n=== NIAH Test Results ===")
            print(f"Context length: {result['context_length']}")
            print(f"Needle text: {result['needle_text']}")
            
            for test in result["tests"]:
                position = test["position"]
                success_rate = test["success_rate"]
                avg_time = test["avg_inference_time"]
                
                print(f"\nPosition {position:.2f}:")
                print(f"  Success rate: {success_rate:.2f}")
                print(f"  Average inference time: {avg_time:.2f}s")
                
                for i, run in enumerate(test["runs"]):
                    if "error" in run:
                        print(f"  Run {i+1}: Error - {run['error']}")
                    else:
                        print(f"  Run {i+1}: {'Success' if run['success'] else 'Failure'}")
                        print(f"    Response: {run['response']}")
                        print(f"    Inference time: {run['inference_time']:.2f}s")
                
    else:  # Custom text test
        if args.input_file:
            text = tester.load_text_from_file(args.input_file)
        elif args.input_text:
            text = args.input_text
        else:
            parser.error("For custom text test, either --input-file or --input-text must be provided")
        
        result = tester.run_custom_text_test(
            text=text,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print("\n=== Custom Text Test Results ===")
            print(f"Context length: {result['context_length']} tokens")
            print(f"Inference time: {result['inference_time']:.2f}s")
            print(f"Tokens per second: {result['tokens_per_second']:.2f}")
            print(f"\nResponse: {result['response']}")


if __name__ == "__main__":
    main()
