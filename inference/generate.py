import os
import json
import time
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs, ModelArgsSmall, ModelArgsMedium, ModelArgsWithMTP


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


@torch.inference_mode()
def generate_speculative(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    spec_length: int = 4  # Number of tokens to predict speculatively
) -> List[List[int]]:
    """
    Generate tokens with speculative decoding using the MTP module.
    
    Args:
        model (Transformer): The transformer model with MTP module.
        prompt_tokens (List[List[int]]): List of token IDs for each prompt.
        max_new_tokens (int): Maximum number of new tokens to generate.
        eos_id (int): End of sequence token ID.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        spec_length (int, optional): Number of tokens to predict speculatively. Defaults to 4.
        
    Returns:
        List[List[int]]: Generated token IDs for each prompt.
    """
    assert hasattr(model, 'mtp_layers') and len(model.mtp_layers) > 0, "Model does not have MTP layers"
    
    B = len(prompt_tokens)
    L = max([len(x) for x in prompt_tokens])
    
    tokens = torch.full((B, L), 0, dtype=torch.long, device="cuda")
    for i, x in enumerate(prompt_tokens):
        tokens[i, :len(x)] = torch.tensor(x, dtype=torch.long, device="cuda")
    
    eos_reached = torch.zeros(B, dtype=torch.bool, device="cuda")
    input_pos = 0
    
    for _ in range(max_new_tokens):
        if eos_reached.all():
            break
            
        if input_pos == 0:
            logits = model(tokens)
        else:
            
            candidates = []
            curr_tokens = tokens
            curr_pos = input_pos
            
            for i in range(min(spec_length, max_new_tokens - len(candidates))):
                h = model.embed(curr_tokens[:, -1:])
                freqs_cis = model.freqs_cis[curr_pos:curr_pos+1]
                
                for mtp_layer in model.mtp_layers:
                    h = mtp_layer(h, curr_pos, freqs_cis, None)
                
                h = model.norm(h)[:, -1]
                mtp_logits = model.head(h)
                
                next_token = sample(mtp_logits, temperature)
                candidates.append(next_token.item())
                
                curr_tokens = torch.cat([curr_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                curr_pos += 1
            
            verification_tokens = torch.cat([tokens, torch.tensor([candidates], dtype=torch.long, device="cuda")], dim=1)
            verification_logits = model(verification_tokens)
            
            accepted_tokens = []
            for i, candidate in enumerate(candidates):
                main_probs = torch.softmax(verification_logits[i], dim=-1)
                
                if torch.argmax(main_probs).item() == candidate:
                    accepted_tokens.append(candidate)
                else:
                    break
            
            if accepted_tokens:
                for token in accepted_tokens:
                    tokens = torch.cat([tokens, torch.tensor([[token]], dtype=torch.long, device="cuda")], dim=1)
                    
                    if token == eos_id:
                        eos_reached = torch.ones(1, dtype=torch.bool, device="cuda")
                        break
                        
                logits = verification_logits
            else:
                next_token = sample(verification_logits, temperature)
                tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                logits = verification_logits
        
        input_pos = tokens.size(1)
    
    output = []
    for i, row in enumerate(tokens.tolist()):
        if eos_id != -1:
            try:
                end = row.index(eos_id, len(prompt_tokens[i]))
                output.append(row[:end])
            except ValueError:
                output.append(row)
        else:
            output.append(row)
    
    return output


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    use_speculative: bool = False,
    spec_length: int = 4,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            t0 = time.time()
            if use_speculative:
                completion_tokens = generate_speculative(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, spec_length)
            else:
                completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            print(f"Time: {time.time() - t0:.2f}s, Tokens: {len(completion_tokens[0])}, Speed: {len(completion_tokens[0]) / (time.time() - t0):.2f} tok/s")
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--use-speculative", action="store_true", help="Use speculative decoding with MTP module")
    parser.add_argument("--spec-length", type=int, default=4, help="Number of tokens to predict speculatively")
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.use_speculative, args.spec_length)
