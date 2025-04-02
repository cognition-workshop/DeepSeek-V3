import os
import sys
import json
import torch
from typing import Dict, Any, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.model import Transformer, ModelArgs
from inference.generate import generate
from transformers import AutoTokenizer

class DeepSeekWrapper:
    """Wrapper for DeepSeek-V3 model to perform summarization and sentiment analysis."""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the DeepSeek model wrapper.
        
        Args:
            model_path: Path to the model weights
            config_path: Path to the model configuration
        """
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(0)  # Assuming we're using a single GPU
        
        with open(config_path) as f:
            args = ModelArgs(**json.load(f))
        
        self.model = Transformer(args).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        world_size = 1  # For single GPU setup
        rank = 0
        from safetensors.torch import load_model
        load_model(self.model, os.path.join(model_path, f"model{rank}-mp{world_size}.safetensors"))
        
        self.summarization_template = [
            {"role": "user", "content": "Please summarize the following customer support ticket in a concise paragraph (maximum 3 sentences):\n\n{ticket}"}
        ]
        
        self.sentiment_template = [
            {"role": "user", "content": "Analyze the sentiment of the following customer support ticket and provide a score from 1 to 10, where 1 is extremely negative and 10 is extremely positive. Only provide the numerical score without any explanation:\n\n{ticket}"}
        ]
    
    @torch.inference_mode()
    def process_ticket(self, ticket: str) -> Tuple[str, int]:
        """
        Process a customer support ticket to generate a summary and sentiment score.
        
        Args:
            ticket: The customer support ticket text
            
        Returns:
            Tuple containing the summary and sentiment score
        """
        summary = self._generate_text(self.summarization_template, ticket, max_new_tokens=150)
        
        sentiment_result = self._generate_text(self.sentiment_template, ticket, max_new_tokens=10)
        try:
            sentiment_score = int(sentiment_result.strip())
            sentiment_score = max(1, min(10, sentiment_score))
        except ValueError:
            sentiment_score = 5
            
        return summary, sentiment_score
    
    def _generate_text(self, template: List[Dict[str, str]], ticket: str, max_new_tokens: int = 100) -> str:
        """
        Generate text using the DeepSeek model.
        
        Args:
            template: The prompt template to use
            ticket: The customer support ticket to process
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        formatted_template = [
            {"role": item["role"], "content": item["content"].format(ticket=ticket)}
            for item in template
        ]
        
        prompt_tokens = self.tokenizer.apply_chat_template(
            formatted_template, 
            add_generation_prompt=True
        )
        
        completion_tokens = generate(
            self.model, 
            [prompt_tokens], 
            max_new_tokens=max_new_tokens, 
            eos_id=self.tokenizer.eos_token_id, 
            temperature=0.7  # Adjust temperature as needed
        )
        
        completion = self.tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        
        return completion.strip()
