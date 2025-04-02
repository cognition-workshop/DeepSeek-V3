import os
import json
import torch
import argparse
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from attention_extractor import AttentionExtractor

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['JSON_SORT_KEYS'] = False

model = None
tokenizer = None
attention_extractor = None


def initialize_model(ckpt_path, config_path):
    """Initialize the model and tokenizer"""
    global model, tokenizer, attention_extractor
    
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
    
    with torch.device("cuda"):
        model = Transformer(args)
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    
    attention_extractor = AttentionExtractor(model, tokenizer)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/extract_attention', methods=['POST'])
def extract_attention():
    """API endpoint to extract attention patterns"""
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({"error": "No input texts provided"}), 400
    
    results = []
    for text in texts:
        attention_data = attention_extractor.extract_attention(text)
        results.append(attention_data)
    
    return jsonify({"results": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSeek-V3 Attention Visualization Tool')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    initialize_model(args.ckpt_path, args.config)
    app.run(host=args.host, port=args.port, debug=True)
