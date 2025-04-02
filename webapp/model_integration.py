import os
import json
import torch
from typing import List, Dict, Any, Optional
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../inference'))

from model import Transformer, ModelArgs
from transformers import AutoTokenizer
from safetensors.torch import load_model
from generate import generate as model_generate

class ModelWrapper:
    def __init__(self, ckpt_path: str, config_path: str):
        """Initialize the model wrapper with paths to the checkpoint and config."""
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the model and tokenizer."""
        if self.initialized:
            return
            
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
        if world_size > 1:
            import torch.distributed as dist
            dist.init_process_group("nccl")
            
        torch.cuda.set_device(local_rank)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)
        torch.manual_seed(965)
        
        with open(self.config_path) as f:
            args = ModelArgs(**json.load(f))
            
        with torch.device("cuda"):
            self.model = Transformer(args)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt_path)
        
        self.tokenizer.decode(model_generate(self.model, [self.tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
        
        load_model(self.model, os.path.join(self.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        
        self.initialized = True
        
    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a response using the model based on the message history."""
        if not self.initialized:
            self.initialize()
            
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Call initialize() first.")
            
        prompt_tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        completion_tokens = model_generate(
            self.model, 
            [prompt_tokens], 
            max_new_tokens, 
            self.tokenizer.eos_token_id, 
            temperature
        )
        
        return self.tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
