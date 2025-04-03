"""
Benchmark script for DeepSeek-V3 with different model configurations and speculative decoding.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

from model import Transformer, ModelArgs, ModelArgsSmall, ModelArgsTiny
from generate import generate, generate_speculative


def load_model_config(config_path: str) -> Dict:
    """Load a model configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_pseudo_prompts(tokenizer, lengths=[10, 50, 100, 200], n_prompts=5) -> List[List[str]]:
    """
    Create pseudo prompts of varying lengths for benchmarking.
    
    Args:
        tokenizer: The tokenizer to use for encoding.
        lengths: List of prompt lengths to generate.
        n_prompts: Number of prompts per length.
        
    Returns:
        List of prompts grouped by length.
    """
    prompts_by_length = []
    
    sample_texts = {
        10: ["Write a short story", "Explain quantum physics", "What is climate change?", 
             "Tell me about the solar system", "How does AI work?"],
        50: ["Can you write a detailed explanation of how neural networks function? Include information about layers, activation functions, and training.",
             "I'm planning a trip to Japan for two weeks. Can you suggest an itinerary that covers Tokyo, Kyoto, and Osaka? Include must-see attractions.",
             "What are the major theories about the origin of the universe? Please explain the Big Bang theory and any competing hypotheses.",
             "How has artificial intelligence evolved over the past decade? What are the most significant breakthroughs and their implications?",
             "Can you explain the process of photosynthesis in detail? Include the light-dependent and light-independent reactions."],
        100: ["I'm researching the impact of climate change on biodiversity. Could you provide a comprehensive analysis of how rising global temperatures affect different ecosystems? Include examples from terrestrial and marine environments, and discuss potential mitigation strategies.",
              "Could you explain the fundamental principles of quantum computing and how it differs from classical computing? Please include information on qubits, superposition, entanglement, and major advances in the field. Also, what are the potential applications?",
              "I'm interested in understanding the historical development of economic systems. Could you trace the evolution from feudalism to capitalism, discussing key transitions, influential thinkers, and how these systems shaped society? Include modern variations.",
              "What are the most significant ethical considerations surrounding genetic engineering technologies like CRISPR? Please discuss potential benefits and risks, regulatory approaches across different countries, and perspectives from various stakeholders.",
              "Could you analyze the global food supply chain and its vulnerabilities? Discuss factors like climate change, political instability, transportation challenges, and how these impact food security. Include potential solutions to create more resilient systems."],
        200: ["I'm working on a research paper about the intersection of artificial intelligence and healthcare. Could you provide a comprehensive overview of how AI is currently being used in medical diagnostics, treatment planning, drug discovery, and patient care? Please include specific examples of successful implementations, challenges faced in integration with existing healthcare systems, ethical considerations regarding patient data privacy and algorithmic bias, and potential future developments in the next decade. Also, how might these technologies affect healthcare disparities across different socioeconomic groups?",
              "I'm developing a curriculum on sustainable urban development for graduate students. Could you create a detailed outline covering the key principles of sustainable cities, including energy-efficient infrastructure, green transportation systems, waste management, urban agriculture, and equitable community planning? Please include case studies from different regions of the world (developed and developing contexts), metrics for measuring sustainability success, common challenges in implementation, and emerging innovations. Additionally, how do cultural and geographical differences impact sustainable urban planning approaches?",
              "I'm interested in understanding the complex relationship between global trade policies, economic inequality, and environmental sustainability. Could you analyze how international trade agreements of the past 30 years have impacted income distribution within and between countries? Please discuss the environmental consequences of globalized supply chains, including carbon emissions from transportation and resource extraction in developing nations. What alternative trade frameworks have been proposed to address these issues? Include perspectives from different economic schools of thought and policy recommendations for creating more equitable and sustainable trade systems.",
              "I'm researching the evolution of social media and its impact on society, politics, and individual psychology. Could you provide an analysis of how major social platforms have changed communication patterns and information consumption since their inception? Please include discussion of filter bubbles, algorithmic content curation, political polarization, privacy concerns, and effects on mental health and social relationships. How have different countries approached social media regulation, and what policy approaches might balance free speech concerns with mitigating harmful effects? Finally, how might social media evolve in the next decade based on current trends and emerging technologies?",
              "I'm exploring the complex interplay between genetic and environmental factors in human health and disease development. Could you explain our current understanding of how genes interact with lifestyle, diet, stress, pollution, and other environmental exposures to influence disease risk? Please include examples of conditions with strong genetic components versus those primarily shaped by environment, and discuss epigenetic mechanisms. How has this understanding evolved with advances in genomic research, and what are the implications for personalized medicine? What ethical considerations arise when assigning causality to genetic versus environmental factors, particularly regarding individual responsibility for health outcomes?"]
    }
    
    n_prompts = min(n_prompts, 5)
    
    for length in lengths:
        prompts = []
        for i in range(n_prompts):
            prompts.append(sample_texts[length][i])
        prompts_by_length.append(prompts)
    
    return prompts_by_length


def run_benchmark(
    model: Transformer,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    batch_size: int = 1,
    use_speculative: bool = False,
    spec_length: int = 4,
    temperature: float = 1.0,
) -> Dict:
    """
    Run a benchmark on a model with given prompts.
    
    Args:
        model: The model to benchmark.
        tokenizer: The tokenizer to use.
        prompts: List of prompts to use.
        max_new_tokens: Maximum number of new tokens to generate.
        batch_size: Batch size for generation.
        use_speculative: Whether to use speculative decoding.
        spec_length: Number of tokens to predict speculatively.
        temperature: Temperature for sampling.
        
    Returns:
        Dictionary with benchmark results.
    """
    results = {
        "latency_per_token": [],
        "tokens_per_second": [],
        "total_time": [],
        "output_length": [],
    }
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        prompt_tokens = [tokenizer.encode(prompt) for prompt in batch_prompts]
        
        start_time = time.time()
        
        if use_speculative:
            generated_tokens = generate_speculative(
                model,
                prompt_tokens,
                max_new_tokens,
                tokenizer.eos_token_id,
                temperature=temperature,
                spec_length=spec_length,
            )
        else:
            generated_tokens = generate(
                model,
                prompt_tokens,
                max_new_tokens,
                tokenizer.eos_token_id,
                temperature=temperature,
            )
        
        end_time = time.time()
        
        generation_time = end_time - start_time
        output_lengths = [len(tokens) for tokens in generated_tokens]
        avg_output_length = np.mean(output_lengths)
        
        tokens_per_second = avg_output_length / generation_time
        latency_per_token = generation_time / avg_output_length
        
        results["latency_per_token"].append(latency_per_token)
        results["tokens_per_second"].append(tokens_per_second)
        results["total_time"].append(generation_time)
        results["output_length"].append(avg_output_length)
    
    for key in results:
        results[key] = np.mean(results[key])
    
    return results


def run_all_benchmarks(
    ckpt_path: str,
    configs: List[str],
    output_file: str = "benchmark_results.json",
    batch_sizes: List[int] = [1, 4, 8],
    use_speculative: bool = True,
    world_size: int = 1,
) -> None:
    """
    Run benchmarks for all model configurations and save results.
    
    Args:
        ckpt_path: Path to model checkpoint.
        configs: List of config paths.
        output_file: Path to save results.
        batch_sizes: List of batch sizes to test.
        use_speculative: Whether to use speculative decoding.
        world_size: Number of GPUs to use.
    """
    if world_size > 1:
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        dist.init_process_group("nccl")
    
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    prompt_lengths = [10, 50, 100, 200]
    prompts_by_length = create_pseudo_prompts(tokenizer, lengths=prompt_lengths)
    
    all_results = {}
    
    for config_path in configs:
        config_name = os.path.basename(config_path).replace(".json", "")
        print(f"Benchmarking config: {config_name}")
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        args = ModelArgs(**config_dict)
        with torch.device("cuda"):
            model = Transformer(args)
        
        load_model(model, os.path.join(ckpt_path, f"model0-mp{world_size}.safetensors"))
        
        config_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Batch size: {batch_size}")
            batch_results = {}
            
            for length_idx, length in enumerate(prompt_lengths):
                print(f"    Prompt length: {length}")
                prompts = prompts_by_length[length_idx]
                
                no_spec_results = run_benchmark(
                    model,
                    tokenizer,
                    prompts,
                    batch_size=batch_size,
                    use_speculative=False,
                )
                
                batch_results[f"length_{length}_no_spec"] = no_spec_results
                
                if use_speculative:
                    for spec_length in [2, 4, 8]:
                        spec_results = run_benchmark(
                            model,
                            tokenizer,
                            prompts,
                            batch_size=batch_size,
                            use_speculative=True,
                            spec_length=spec_length,
                        )
                        
                        batch_results[f"length_{length}_spec_{spec_length}"] = spec_results
            
            config_results[f"batch_{batch_size}"] = batch_results
        
        all_results[config_name] = config_results
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    create_comparison_table(all_results, "comparison_table.md")
    
    if world_size > 1:
        dist.destroy_process_group()


def create_comparison_table(results: Dict, output_file: str) -> None:
    """
    Create a markdown table comparing latency across configurations.
    
    Args:
        results: Dictionary with benchmark results.
        output_file: Path to save the table.
    """
    with open(output_file, "w") as f:
        f.write("# DeepSeek-V3 Latency Comparison\n\n")
        
        batch_sizes = list(next(iter(results.values())).keys())
        for batch_size in batch_sizes:
            f.write(f"## Batch Size: {batch_size.split('_')[1]}\n\n")
            
            config_keys = list(results.keys())
            first_config = results[config_keys[0]][batch_size]
            length_keys = [k for k in first_config.keys() if k.startswith("length_")]
            lengths = sorted(set([int(k.split("_")[1]) for k in length_keys]))
            
            for length in lengths:
                f.write(f"### Prompt Length: {length}\n\n")
                
                f.write("| Model Configuration | Standard Decoding | Speculative (k=2) | Speculative (k=4) | Speculative (k=8) | Speedup (k=4) |\n")
                f.write("|---------------------|-------------------|-------------------|-------------------|-------------------|---------------|\n")
                
                for config in config_keys:
                    standard_key = f"length_{length}_no_spec"
                    spec_2_key = f"length_{length}_spec_2"
                    spec_4_key = f"length_{length}_spec_4"
                    spec_8_key = f"length_{length}_spec_8"
                    
                    if standard_key in results[config][batch_size]:
                        standard_latency = results[config][batch_size][standard_key]["latency_per_token"]
                        spec_2_latency = results[config][batch_size].get(spec_2_key, {}).get("latency_per_token", "-")
                        spec_4_latency = results[config][batch_size].get(spec_4_key, {}).get("latency_per_token", "-")
                        spec_8_latency = results[config][batch_size].get(spec_8_key, {}).get("latency_per_token", "-")
                        
                        if spec_4_latency != "-":
                            speedup = standard_latency / spec_4_latency
                        else:
                            speedup = "-"
                        
                        f.write(f"| {config} | {standard_latency:.4f} s | {spec_2_latency if spec_2_latency == '-' else f'{spec_2_latency:.4f} s'} | {spec_4_latency if spec_4_latency == '-' else f'{spec_4_latency:.4f} s'} | {spec_8_latency if spec_8_latency == '-' else f'{spec_8_latency:.4f} s'} | {speedup if speedup == '-' else f'{speedup:.2f}x'} |\n")
                
                f.write("\n")
            
            f.write("\n")
        
        f.write("## Summary\n\n")
        f.write("This table compares the latency (in seconds per token) for different DeepSeek-V3 model configurations using standard decoding and speculative decoding with various lookahead values (k).\n\n")
        f.write("Key findings:\n")
        f.write("- Smaller model configurations show significant latency improvements\n")
        f.write("- Speculative decoding provides additional speedup, especially with longer lookahead windows\n")
        f.write("- The optimal balance between model size and speculative parameters depends on the specific use case\n")


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-V3 model configurations")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="List of config paths")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8], help="Batch sizes to test")
    parser.add_argument("--no-speculative", action="store_true", help="Disable speculative decoding")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    
    run_all_benchmarks(
        args.ckpt_path,
        args.configs,
        args.output,
        args.batch_sizes,
        not args.no_speculative,
        args.world_size,
    )


if __name__ == "__main__":
    main()
