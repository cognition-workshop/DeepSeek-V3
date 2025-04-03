import os
import json
import time
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import generate, generate_speculative

def generate_pseudo_prompts(num_prompts: int, lengths: List[int]) -> Dict[int, List[str]]:
    """
    Generate pseudo-prompts of varying lengths for benchmarking.
    
    Args:
        num_prompts (int): Number of prompts to generate for each length
        lengths (List[int]): List of prompt lengths to generate
        
    Returns:
        Dict[int, List[str]]: Dictionary mapping prompt lengths to lists of prompts
    """
    result = {}
    for length in lengths:
        prompts = []
        for i in range(num_prompts):
            words = [f"word{j}" for j in range(length // 5)]  # Assuming average word length of 5
            prompt = "This is a benchmarking prompt: " + " ".join(words)
            prompts.append(prompt)
        result[length] = prompts
    return result

def run_benchmark(
    model_config: str,
    checkpoint_path: str,
    prompts: Dict[int, List[str]],
    batch_sizes: List[int] = [1, 4, 8],
    use_speculative: bool = False,
    spec_length: int = 4,
    max_new_tokens: int = 100,
    temperature: float = 0.0,  # Use greedy decoding for consistent benchmarking
    n_runs: int = 3,  # Number of runs to average over
) -> Dict[str, Any]:
    """
    Run benchmarks on a model configuration and return metrics.
    
    Args:
        model_config (str): Path to model configuration file
        checkpoint_path (str): Path to model checkpoint
        prompts (Dict[int, List[str]]): Dictionary mapping prompt lengths to lists of prompts
        batch_sizes (List[int], optional): Batch sizes to benchmark. Defaults to [1, 4, 8].
        use_speculative (bool, optional): Whether to use speculative decoding. Defaults to False.
        spec_length (int, optional): Number of tokens to predict speculatively. Defaults to 4.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 0.0 (greedy).
        n_runs (int, optional): Number of runs to average over. Defaults to 3.
        
    Returns:
        Dict[str, Any]: Dictionary containing benchmark results
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    
    model_name = os.path.basename(model_config).replace("config_", "").replace(".json", "")
    
    with open(model_config) as f:
        args = ModelArgs(**json.load(f))
        
    with torch.device("cuda"):
        model = Transformer(args)
        
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    try:
        load_model(model, os.path.join(checkpoint_path, f"model{rank}-mp{world_size}.safetensors"))
    except Exception as e:
        if rank == 0:
            print(f"Warning: Could not load checkpoint for {model_name}: {e}")
            print("This is expected for smaller model configurations where weights don't exist.")
            print("Benchmarking will proceed with randomly initialized weights.")
    
    results = {
        "model_name": model_name,
        "parameters": {
            "total": model_parameters_count(model),
            "activated": calculate_activated_params(args),
        },
        "world_size": world_size,
        "speculative": use_speculative,
        "spec_length": spec_length if use_speculative else 0,
        "batch_sizes": {},
    }
    
    for batch_size in batch_sizes:
        if batch_size > args.max_batch_size:
            if rank == 0:
                print(f"Skipping batch size {batch_size} which exceeds model's max_batch_size {args.max_batch_size}")
            continue
            
        batch_results = {}
        
        for prompt_length, prompt_list in prompts.items():
            batch_prompts = prompt_list[:batch_size]
            
            prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in batch_prompts]
            
            latencies = []
            token_throughputs = []
            
            for _ in range(n_runs):
                if use_speculative:
                    generate_speculative(model, prompt_tokens, 10, tokenizer.eos_token_id, temperature, True, spec_length)
                else:
                    generate(model, prompt_tokens, 10, tokenizer.eos_token_id, temperature)
                    
                torch.cuda.empty_cache()
                
                start_time = time.time()
                if use_speculative:
                    completion_tokens = generate_speculative(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature, True, spec_length)
                else:
                    completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
                end_time = time.time()
                
                total_time = end_time - start_time
                total_tokens_generated = sum(len(tokens) for tokens in completion_tokens)
                
                latencies.append(total_time)
                token_throughputs.append(total_tokens_generated / total_time)
            
            avg_latency = sum(latencies) / len(latencies)
            avg_throughput = sum(token_throughputs) / len(token_throughputs)
            
            batch_results[prompt_length] = {
                "latency_seconds": avg_latency,
                "tokens_per_second": avg_throughput,
                "tokens_per_second_per_batch": avg_throughput / batch_size,
            }
            
            if rank == 0:
                print(f"Model: {model_name}, Batch: {batch_size}, Prompt Length: {prompt_length}, " +
                      f"Speculative: {use_speculative}, Latency: {avg_latency:.4f}s, " +
                      f"Throughput: {avg_throughput:.2f} tokens/sec")
        
        results["batch_sizes"][batch_size] = batch_results
    
    return results

def model_parameters_count(model: torch.nn.Module) -> int:
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def calculate_activated_params(args: ModelArgs) -> int:
    """
    Calculate the number of activated parameters based on the model configuration.
    
    This is an approximation since we don't have the exact formula used in the paper.
    """
    embed_params = args.vocab_size * args.dim
    norm_params = args.n_layers * args.dim * 2  # Assuming 2 RMSNorm per layer
    head_params = args.dim * args.vocab_size
    
    attn_params = args.dim * (args.dim * 3)  # Q, K, V projections
    
    dense_ffn_params = args.n_dense_layers * (args.dim * args.inter_dim * 3)  # 3 matrices in MLP
    
    moe_layers = args.n_layers - args.n_dense_layers
    moe_params_per_layer = args.n_activated_experts * (args.dim * args.moe_inter_dim * 3)
    moe_params = moe_layers * moe_params_per_layer
    
    total_activated = embed_params + norm_params + head_params + attn_params * args.n_layers + dense_ffn_params + moe_params
    
    return total_activated

def create_comparison_table(results: List[Dict[str, Any]]) -> str:
    """
    Create a formatted comparison table from benchmark results.
    
    Args:
        results (List[Dict[str, Any]]): List of benchmark result dictionaries
        
    Returns:
        str: Formatted markdown table with comparison results
    """
    grouped_results = {
        False: [],  # non-speculative
        True: [],   # speculative
    }
    
    for result in results:
        grouped_results[result["speculative"]].append(result)
    
    for key in grouped_results:
        grouped_results[key] = sorted(grouped_results[key], 
                                     key=lambda r: r["parameters"]["total"])
    
    table = "## Latency Comparison Table\n\n"
    table += "### Model Configurations\n\n"
    table += "| Model | Total Parameters | Activated Parameters | Layers | Heads | Experts | Activated Experts |\n"
    table += "|-------|-----------------|----------------------|--------|-------|---------|-------------------|\n"
    
    for result in grouped_results[False]:  # Use non-speculative group for config info
        model_name = result["model_name"]
        total_params = result["parameters"]["total"] / 1e9  # Convert to billions
        activated_params = result["parameters"]["activated"] / 1e9  # Convert to billions
        
        if model_name == "671B":
            layers, heads, experts, activated = 61, 128, 256, 8
        elif model_name == "236B":
            layers, heads, experts, activated = 60, 128, 160, 6
        elif model_name == "16B":
            layers, heads, experts, activated = 27, 16, 64, 6
        elif model_name == "small":
            layers, heads, experts, activated = 12, 8, 32, 4
        elif model_name == "tiny":
            layers, heads, experts, activated = 6, 4, 16, 2
        else:
            layers, heads, experts, activated = "?", "?", "?", "?"
            
        table += f"| {model_name} | {total_params:.2f}B | {activated_params:.2f}B | {layers} | {heads} | {experts} | {activated} |\n"
    
    batch_sizes = set()
    prompt_lengths = set()
    
    for result in results:
        batch_sizes.update(result["batch_sizes"].keys())
        for batch_size, batch_data in result["batch_sizes"].items():
            prompt_lengths.update(batch_data.keys())
    
    batch_sizes = sorted(batch_sizes)
    prompt_lengths = sorted(prompt_lengths)
    
    for batch_size in batch_sizes:
        table += f"\n### Batch Size: {batch_size}\n\n"
        
        table += "#### Latency (seconds)\n\n"
        table += "| Model | " + " | ".join([f"Prompt Length {length}" for length in prompt_lengths]) + " | Speculative |\n"
        table += "|-------| " + " | ".join(["---------------" for _ in prompt_lengths]) + " |------------|\n"
        
        for result in grouped_results[False]:
            model_name = result["model_name"]
            if batch_size not in result["batch_sizes"]:
                continue
                
            row = f"| {model_name} | "
            for length in prompt_lengths:
                if length in result["batch_sizes"][batch_size]:
                    latency = result["batch_sizes"][batch_size][length]["latency_seconds"]
                    row += f"{latency:.4f} | "
                else:
                    row += "N/A | "
            row += "No |"
            table += row + "\n"
        
        for result in grouped_results[True]:
            model_name = result["model_name"]
            if batch_size not in result["batch_sizes"]:
                continue
                
            row = f"| {model_name} (spec) | "
            for length in prompt_lengths:
                if length in result["batch_sizes"][batch_size]:
                    latency = result["batch_sizes"][batch_size][length]["latency_seconds"]
                    row += f"{latency:.4f} | "
                else:
                    row += "N/A | "
            row += f"Yes ({result['spec_length']}) |"
            table += row + "\n"
        
        table += "\n#### Throughput (tokens/second)\n\n"
        table += "| Model | " + " | ".join([f"Prompt Length {length}" for length in prompt_lengths]) + " | Speculative |\n"
        table += "|-------| " + " | ".join(["---------------" for _ in prompt_lengths]) + " |------------|\n"
        
        for result in grouped_results[False]:
            model_name = result["model_name"]
            if batch_size not in result["batch_sizes"]:
                continue
                
            row = f"| {model_name} | "
            for length in prompt_lengths:
                if length in result["batch_sizes"][batch_size]:
                    throughput = result["batch_sizes"][batch_size][length]["tokens_per_second"]
                    row += f"{throughput:.2f} | "
                else:
                    row += "N/A | "
            row += "No |"
            table += row + "\n"
        
        for result in grouped_results[True]:
            model_name = result["model_name"]
            if batch_size not in result["batch_sizes"]:
                continue
                
            row = f"| {model_name} (spec) | "
            for length in prompt_lengths:
                if length in result["batch_sizes"][batch_size]:
                    throughput = result["batch_sizes"][batch_size][length]["tokens_per_second"]
                    row += f"{throughput:.2f} | "
                else:
                    row += "N/A | "
            row += f"Yes ({result['spec_length']}) |"
            table += row + "\n"
    
    table += "\n### Speedup from Speculative Decoding\n\n"
    table += "| Model | " + " | ".join([f"Batch Size {bs}" for bs in batch_sizes]) + " |\n"
    table += "|-------| " + " | ".join(["------------" for _ in batch_sizes]) + " |\n"
    
    for model_name in set(r["model_name"] for r in results):
        row = f"| {model_name} | "
        
        for batch_size in batch_sizes:
            non_spec = next((r for r in grouped_results[False] if r["model_name"] == model_name and batch_size in r["batch_sizes"]), None)
            spec = next((r for r in grouped_results[True] if r["model_name"] == model_name and batch_size in r["batch_sizes"]), None)
            
            if non_spec and spec and batch_size in non_spec["batch_sizes"] and batch_size in spec["batch_sizes"]:
                speedups = []
                for length in prompt_lengths:
                    if length in non_spec["batch_sizes"][batch_size] and length in spec["batch_sizes"][batch_size]:
                        non_spec_throughput = non_spec["batch_sizes"][batch_size][length]["tokens_per_second"]
                        spec_throughput = spec["batch_sizes"][batch_size][length]["tokens_per_second"]
                        speedup = spec_throughput / non_spec_throughput if non_spec_throughput > 0 else 0
                        speedups.append(speedup)
                
                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    row += f"{avg_speedup:.2f}x | "
                else:
                    row += "N/A | "
            else:
                row += "N/A | "
        
        table += row + "\n"
    
    return table

def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V3 model configurations")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="List of model configuration files to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to benchmark")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[50, 200, 500], help="Prompt lengths to benchmark")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts per length")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--use-speculative", action="store_true", help="Whether to use speculative decoding")
    parser.add_argument("--spec-length", type=int, default=4, help="Number of tokens to predict speculatively")
    parser.add_argument("--output-file", type=str, default="benchmark_results.json", help="Output file for benchmark results")
    parser.add_argument("--table-file", type=str, default="benchmark_table.md", help="Output file for benchmark table")
    args = parser.parse_args()
    
    prompts = generate_pseudo_prompts(args.num_prompts, args.prompt_lengths)
    
    results = []
    
    for config in args.configs:
        result = run_benchmark(
            model_config=config,
            checkpoint_path=args.ckpt_path,
            prompts=prompts,
            batch_sizes=args.batch_sizes,
            use_speculative=False,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(result)
        
        if args.use_speculative:
            result = run_benchmark(
                model_config=config,
                checkpoint_path=args.ckpt_path,
                prompts=prompts,
                batch_sizes=args.batch_sizes,
                use_speculative=True,
                spec_length=args.spec_length,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(result)
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    table = create_comparison_table(results)
    with open(args.table_file, "w") as f:
        f.write(table)
    
    print(f"Benchmark results saved to {args.output_file}")
    print(f"Comparison table saved to {args.table_file}")

if __name__ == "__main__":
    main()
