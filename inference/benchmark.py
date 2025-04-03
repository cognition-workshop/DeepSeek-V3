import os
import json
import time
import argparse
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from model import Transformer, ModelArgs, ModelArgsSmall, ModelArgsTiny
from generate import generate, generate_speculative


def create_pseudo_prompts(n_prompts: int, lengths: List[int]) -> Dict[int, List[str]]:
    """
    Create pseudo prompts of varying lengths for benchmarking.
    
    Args:
        n_prompts (int): Number of prompts to generate per length.
        lengths (List[int]): List of prompt lengths to generate.
        
    Returns:
        Dict[int, List[str]]: Dictionary mapping prompt length to list of prompts.
    """
    prompts_by_length = {}
    for length in lengths:
        prompts = []
        for i in range(n_prompts):
            prompt = f"This is a benchmark prompt {i} with target length {length}. " * (length // 10)
            prompts.append(prompt)
        prompts_by_length[length] = prompts
    return prompts_by_length


def run_benchmark(
    config_path: str,
    ckpt_path: str,
    prompts_by_length: Dict[int, List[str]],
    batch_sizes: List[int],
    max_new_tokens: int,
    temperature: float,
    use_speculative: bool,
    spec_length: int,
) -> Dict[str, Any]:
    """
    Run benchmarks for the given configuration and prompts.
    
    Args:
        config_path (str): Path to the model configuration file.
        ckpt_path (str): Path to the model checkpoint.
        prompts_by_length (Dict[int, List[str]]): Dictionary of prompts by length.
        batch_sizes (List[int]): List of batch sizes to test.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Temperature for sampling.
        use_speculative (bool): Whether to use speculative decoding.
        spec_length (int): Number of tokens to predict speculatively.
        
    Returns:
        Dict[str, Any]: Dictionary containing benchmark results.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    
    with open(config_path) as f:
        config = json.load(f)
        model_args = ModelArgs(**config)
    
    print(f"Benchmarking model with config: {os.path.basename(config_path)}")
    print(f"Model parameters: n_layers={model_args.n_layers}, dim={model_args.dim}, n_heads={model_args.n_heads}")
    
    with torch.device("cuda"):
        model = Transformer(model_args)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    if os.path.exists(os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")):
        from safetensors.torch import load_model
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    else:
        print(f"Warning: Could not find model weights at {os.path.join(ckpt_path, f'model{rank}-mp{world_size}.safetensors')}")
        print("Running with uninitialized weights for benchmarking purposes.")
    
    results = {"config": os.path.basename(config_path), "use_speculative": use_speculative}
    for batch_size in batch_sizes:
        if batch_size > model_args.max_batch_size:
            print(f"Skipping batch size {batch_size} as it exceeds model's max_batch_size ({model_args.max_batch_size})")
            continue
        
        batch_results = {}
        for prompt_length, prompts in prompts_by_length.items():
            batch_prompts = prompts[:batch_size]
            
            prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in batch_prompts]
            
            start_time = time.time()
            if use_speculative:
                completion_tokens = generate_speculative(
                    model, 
                    prompt_tokens, 
                    max_new_tokens, 
                    tokenizer.eos_token_id, 
                    temperature,
                    True,
                    spec_length
                )
            else:
                completion_tokens = generate(
                    model, 
                    prompt_tokens, 
                    max_new_tokens, 
                    tokenizer.eos_token_id, 
                    temperature
                )
            end_time = time.time()
            
            total_tokens = sum(len(tokens) for tokens in completion_tokens)
            elapsed_time = end_time - start_time
            tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
            latency_per_token = elapsed_time / total_tokens if total_tokens > 0 else 0
            
            batch_results[prompt_length] = {
                "elapsed_time": elapsed_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "latency_per_token": latency_per_token,
            }
            
            print(f"Batch size: {batch_size}, Prompt length: {prompt_length}, Time: {elapsed_time:.2f}s, Tokens/sec: {tokens_per_second:.2f}")
        
        results[f"batch_size_{batch_size}"] = batch_results
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    
    return results


def main():
    """
    Main function for running benchmarks.
    """
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V3 model configurations")
    parser.add_argument("--config-paths", type=str, nargs="+", required=True, help="Paths to model configuration files")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024], help="Prompt lengths to test")
    parser.add_argument("--n-prompts", type=int, default=10, help="Number of prompts per length")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--use-speculative", action="store_true", help="Use speculative decoding")
    parser.add_argument("--spec-length", type=int, default=4, help="Number of tokens to predict speculatively")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json", help="Path to save benchmark results")
    
    args = parser.parse_args()
    
    prompts_by_length = create_pseudo_prompts(args.n_prompts, args.prompt_lengths)
    
    all_results = {}
    for config_path in args.config_paths:
        results = run_benchmark(
            config_path,
            args.ckpt_path,
            prompts_by_length,
            args.batch_sizes,
            args.max_new_tokens,
            args.temperature,
            args.use_speculative,
            args.spec_length
        )
        
        config_name = os.path.basename(config_path).replace(".json", "")
        all_results[config_name] = results
    
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Benchmark results saved to {args.output_file}")
    
    print("\nLatency Comparison Table:")
    print("-" * 80)
    print(f"{'Config':<15} | {'Batch Size':<10} | {'Prompt Length':<15} | {'Latency (ms/token)':<20} | {'Tokens/sec':<15}")
    print("-" * 80)
    
    for config_name, results in all_results.items():
        for batch_key, batch_results in results.items():
            if batch_key.startswith("batch_size_"):
                batch_size = batch_key.replace("batch_size_", "")
                for prompt_length, metrics in batch_results.items():
                    print(f"{config_name:<15} | {batch_size:<10} | {prompt_length:<15} | "
                          f"{metrics['latency_per_token'] * 1000:.2f} ms/token | {metrics['tokens_per_second']:.2f}")
    
    print("-" * 80)


if __name__ == "__main__":
    main()
