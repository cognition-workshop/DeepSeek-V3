import os
import json
from argparse import ArgumentParser
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import generate as original_generate
from rag_integrator import RAGIntegrator

rag = RAGIntegrator()

def generate_with_rag(
    model: Transformer,
    tokenizer,
    query: str,
    chat_history: Optional[List[Dict]] = None,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    pdf_path: Optional[str] = None
) -> str:
    """
    Generate a response using the model with RAG enhancement.

    Args:
        model (Transformer): The transformer model.
        tokenizer: The tokenizer.
        query (str): User query.
        chat_history (List[Dict], optional): Previous chat history. Defaults to None.
        max_new_tokens (int, optional): Maximum tokens to generate. Defaults to 500.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
        pdf_path (str, optional): Path to the PDF file for context. Defaults to None.

    Returns:
        str: Generated response.
    """
    if chat_history is None:
        chat_history = []
    
    if pdf_path and pdf_path != rag.current_pdf:
        rag.process_pdf(pdf_path)
        
    context = ""
    if rag.current_pdf:
        context = rag.retrieve_context(query)
    
    messages = list(chat_history)  # Create a copy
    
    if context:
        system_message = {
            "role": "system", 
            "content": f"You are a helpful assistant. Use the following information from the PDF to answer the user's question. If the information is not in the provided context, say you don't know: \n\n{context}"
        }
        
        if messages and messages[0]["role"] == "system":
            messages[0] = system_message
        else:
            messages.insert(0, system_message)
    
    messages.append({"role": "user", "content": query})
    
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    completion_tokens = original_generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    
    return completion

def main():
    """
    Main function for the RAG-enhanced text generation script.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--pdf-path", type=str, help="Path to the PDF file for context")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this process")
    args = parser.parse_args()
    
    if args.world_size > 1:
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config_path) as f:
        model_args = ModelArgs(**json.load(f))
    
    with torch.device(device):
        model = Transformer(model_args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    
    load_model(model, os.path.join(args.ckpt_path, f"model{args.rank}-mp{args.world_size}.safetensors"))
    
    if args.pdf_path:
        print(f"Processing PDF: {args.pdf_path}")
        success = rag.process_pdf(args.pdf_path)
        if success:
            print("PDF processed successfully")
        else:
            print("Error processing PDF")
    
    if args.interactive:
        chat_history = []
        
        print("RAG-enhanced DeepSeek Chat. Type 'exit' to quit.")
        while True:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break
                
            response = generate_with_rag(
                model, 
                tokenizer, 
                user_input, 
                chat_history, 
                args.max_new_tokens, 
                args.temperature, 
                args.pdf_path
            )
            
            print(f"\nAssistant: {response}")
            
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
    
    else:
        user_input = input("Enter your query: ")
        response = generate_with_rag(
            model, 
            tokenizer, 
            user_input, 
            [], 
            args.max_new_tokens, 
            args.temperature, 
            args.pdf_path
        )
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
