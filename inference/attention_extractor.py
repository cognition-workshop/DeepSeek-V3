import torch
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer

from model import Transformer, ModelArgs


class AttentionExtractor:
    """
    Utility class for extracting attention patterns from DeepSeek-V3 model.
    """
    def __init__(self, model: Transformer, tokenizer: AutoTokenizer):
        """
        Initialize the attention extractor.
        
        Args:
            model (Transformer): The DeepSeek-V3 model.
            tokenizer (AutoTokenizer): The tokenizer for the model.
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def extract_attention(self, text: str) -> Dict[str, Any]:
        """
        Extract attention patterns for the given text.
        
        Args:
            text (str): The input text.
            
        Returns:
            Dict[str, Any]: Dictionary containing tokens and attention weights.
        """
        messages = [{"role": "user", "content": text}]
        tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        token_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        
        with torch.inference_mode():
            _, attention_weights = self.model(token_ids, return_attention=True)
        
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)
        
        processed_weights = []
        for layer_idx, layer_weights in enumerate(attention_weights):
            layer_weights = layer_weights[0].cpu().numpy()
            processed_weights.append({
                "layer": layer_idx,
                "weights": layer_weights
            })
        
        return {
            "tokens": token_strs,
            "attention_weights": processed_weights
        }
