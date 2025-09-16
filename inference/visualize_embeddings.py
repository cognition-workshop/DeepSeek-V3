#!/usr/bin/env python3
"""
DeepSeek-V3 Embedding Visualization Tool

This script provides an interactive web interface for visualizing DeepSeek-V3 embeddings
using dimensionality reduction techniques like t-SNE, UMAP, and PCA.

Usage:
    python visualize_embeddings.py --ckpt-path /path/to/model --config configs/config_16B.json
    
Example:
    python visualize_embeddings.py --ckpt-path ./model_weights --config configs/config_16B.json --port 5000
"""

import os
import sys
import json
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.web_app import EmbeddingVisualizer, app


def main():
    """Main entry point for the embedding visualization tool."""
    parser = ArgumentParser(description="Visualize DeepSeek-V3 embeddings")
    parser.add_argument("--ckpt-path", type=str, required=True, 
                       help="Path to model checkpoint directory")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to model configuration file")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port for web interface (default: 5000)")
    parser.add_argument("--host", type=str, default="localhost", 
                       help="Host for web interface (default: localhost)")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint path '{args.ckpt_path}' does not exist")
        sys.exit(1)
        
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        sys.exit(1)
    
    try:
        print("Initializing DeepSeek-V3 Embedding Visualizer...")
        print(f"Model path: {args.ckpt_path}")
        print(f"Config: {args.config}")
        
        visualizer = EmbeddingVisualizer(args.ckpt_path, args.config)
        app.config['visualizer'] = visualizer
        
        print(f"\nStarting web server...")
        print(f"Open your browser to http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop the server")
        
        app.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
