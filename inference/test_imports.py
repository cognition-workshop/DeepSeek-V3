#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all visualization module imports."""
    try:
        from visualization.embedding_extractor import EmbeddingExtractor
        print("✓ EmbeddingExtractor import successful")
        
        from visualization.dimensionality_reduction import DimensionalityReducer
        print("✓ DimensionalityReducer import successful")
        
        from visualization.web_app import EmbeddingVisualizer, app
        print("✓ Web app imports successful")
        
        from model import ModelArgs, Transformer
        print("✓ Model imports successful")
        
        import json
        with open('configs/config_16B.json') as f:
            args = ModelArgs(**json.load(f))
        print(f"✓ Model config loaded: vocab_size={args.vocab_size}, dim={args.dim}")
        
        print("\n✅ All imports and basic functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
