import os
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs, ModelArgsSmall, ModelArgsMedium, ModelArgsWithMTP
from generate import generate, generate_speculative


def load_model_config(config_path: str) -> Dict:
    """Load model configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_model(config_path: str, model_size: str = "full") -> Transformer:
    """Create a model instance based on configuration."""
    config = load_model_config(config_path)
    
    if model_size == "small":
        args = ModelArgsSmall()
    elif model_size == "medium":
        args = ModelArgsMedium()
    else:  # "full"
        args = ModelArgs()
    
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)
    
    if "num_nextn_predict_layers" in config:
        args = ModelArgsWithMTP()
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    return Transformer(args)


def get_sample_prompts(n_prompts: int = 10, avg_length: int = 100) -> List[str]:
    """Generate sample prompts of varying lengths for benchmarking."""
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a poem about artificial intelligence.",
        "What are the key differences between Python 2 and Python 3?",
        "Describe the process of photosynthesis in detail.",
        "How do neural networks learn?",
        "What are the implications of quantum computing for cryptography?",
        "Explain how blockchain technology works.",
        "What are the ethical considerations in artificial intelligence development?",
        "Describe the water cycle and its importance for life on Earth.",
        "What is the significance of the Higgs boson discovery?"
    ]
    
    while len(prompts) < n_prompts:
        prompts.extend(prompts)
    
    return prompts[:n_prompts]


def run_benchmark(
    model: Transformer,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    use_speculative: bool = False,
    spec_length: int = 4,
    batch_size: int = 1
) -> Dict:
    """Run benchmarking on given model and prompts."""
    results = {
        "total_time": 0,
        "total_tokens_generated": 0,
        "tokens_per_second": 0,
        "latency_per_token": 0,
        "prompt_processing_time": 0,
        "generation_time": 0
    }
    
    all_prompt_tokens = []
    t0 = time.time()
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        all_prompt_tokens.append(tokens)
    results["prompt_processing_time"] = time.time() - t0
    
    t0 = time.time()
    for i in range(0, len(all_prompt_tokens), batch_size):
        batch_tokens = all_prompt_tokens[i:i+batch_size]
        if use_speculative:
            outputs = generate_speculative(
                model, batch_tokens, max_new_tokens, 
                tokenizer.eos_token_id, 1.0, spec_length
            )
        else:
            outputs = generate(
                model, batch_tokens, max_new_tokens, 
                tokenizer.eos_token_id, 1.0
            )
        
        for j, output in enumerate(outputs):
            results["total_tokens_generated"] += len(output) - len(batch_tokens[j])
    
    results["generation_time"] = time.time() - t0
    results["total_time"] = results["prompt_processing_time"] + results["generation_time"]
    
    if results["total_tokens_generated"] > 0:
        results["tokens_per_second"] = results["total_tokens_generated"] / results["generation_time"]
        results["latency_per_token"] = results["generation_time"] / results["total_tokens_generated"]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V3 model configurations")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-sizes", type=str, default="full,medium,small", help="Comma-separated list of model sizes to benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8", help="Comma-separated list of batch sizes to benchmark")
    parser.add_argument("--n-prompts", type=int, default=10, help="Number of prompts to use for benchmarking")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max number of new tokens to generate")
    parser.add_argument("--use-speculative", action="store_true", help="Use speculative decoding")
    parser.add_argument("--spec-length", type=int, default=4, help="Number of tokens to predict speculatively")
    args = parser.parse_args()
    
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    
    prompts = get_sample_prompts(args.n_prompts)
    
    model_sizes = args.model_sizes.split(",")
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    benchmark_results = {}
    
    for model_size in model_sizes:
        if model_size == "small":
            config_path = os.path.join(os.path.dirname(args.ckpt_path), "configs/config_small.json")
        elif model_size == "medium":
            config_path = os.path.join(os.path.dirname(args.ckpt_path), "configs/config_medium.json")
        else:  # "full"
            config_path = os.path.join(os.path.dirname(args.ckpt_path), "configs/config_671B.json")
        
        model = create_model(config_path, model_size)
        load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        
        benchmark_results[model_size] = {}
        
        for use_spec in [False, True] if args.use_speculative else [False]:
            spec_str = "with_speculative" if use_spec else "standard"
            benchmark_results[model_size][spec_str] = {}
            
            for batch_size in batch_sizes:
                print(f"Benchmarking {model_size} model {spec_str} with batch size {batch_size}...")
                results = run_benchmark(
                    model, tokenizer, prompts, args.max_new_tokens,
                    use_spec, args.spec_length, batch_size
                )
                benchmark_results[model_size][spec_str][f"batch_{batch_size}"] = results
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model Size':<15} {'Decoding':<15} {'Batch':<8} {'Tokens/sec':<12} {'Latency/tok (ms)':<15}")
    print("-"*80)
    
    for model_size in model_sizes:
        for decoding in benchmark_results[model_size]:
            for batch_key, results in benchmark_results[model_size][decoding].items():
                batch = batch_key.split("_")[1]
                print(f"{model_size:<15} {decoding:<15} {batch:<8} {results['tokens_per_second']:<12.2f} {results['latency_per_token']*1000:<15.2f}")
    
    print("="*80)
    
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()
