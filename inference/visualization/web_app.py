from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.utils
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.embedding_extractor import EmbeddingExtractor
from visualization.dimensionality_reduction import DimensionalityReducer
from model import Transformer, ModelArgs
from transformers import AutoTokenizer
from safetensors.torch import load_model
import torch

app = Flask(__name__, template_folder='templates')


class EmbeddingVisualizer:
    """
    Main class for embedding visualization functionality.
    
    Handles model loading, embedding extraction, dimensionality reduction,
    and creation of interactive visualizations.
    """
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        
        with open(config_path) as f:
            self.args = ModelArgs(**json.load(f))
        
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            self.model = Transformer(self.args)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.demo_mode = False
        except Exception:
            print("Warning: Could not load tokenizer, running in demo mode")
            self.tokenizer = None
            self.demo_mode = True
        
        model_file = os.path.join(model_path, "model0-mp1.safetensors")
        if os.path.exists(model_file):
            load_model(self.model, model_file)
        
        self.extractor = EmbeddingExtractor(self.model)
        self.reducer = DimensionalityReducer()
        
    def create_visualization(self, method: str = 'tsne', n_samples: int = 1000, 
                           n_components: int = 2) -> str:
        """
        Create embedding visualization.
        
        Args:
            method: Dimensionality reduction method ('tsne', 'umap', 'pca')
            n_samples: Number of samples to visualize
            n_components: Number of dimensions (2 or 3)
            
        Returns:
            JSON string of Plotly figure
        """
        try:
            embeddings, token_ids = self.extractor.extract_embeddings()
            
            if len(token_ids) > n_samples:
                indices = np.random.choice(len(token_ids), n_samples, replace=False)
                embeddings = embeddings[indices]
                token_ids = [token_ids[i] for i in indices]
            
            if method == 'tsne':
                reduced = self.reducer.reduce_tsne(embeddings, n_components=n_components)
            elif method == 'umap':
                reduced = self.reducer.reduce_umap(embeddings, n_components=n_components)
            else:
                reduced = self.reducer.reduce_pca(embeddings, n_components=n_components)
            
            tokens = []
            for tid in token_ids:
                if self.tokenizer and not self.demo_mode:
                    try:
                        token = self.tokenizer.decode([tid])
                        tokens.append(token if token.strip() else f"<token_{tid}>")
                    except:
                        tokens.append(f"<token_{tid}>")
                else:
                    tokens.append(f"<token_{tid}>")
            
            if n_components == 2:
                fig = go.Figure(data=go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode='markers',
                    text=tokens,
                    hovertemplate='Token: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                    marker=dict(
                        size=6,
                        opacity=0.7,
                        color=np.arange(len(tokens)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Token Index")
                    )
                ))
                
                fig.update_layout(
                    title=f'DeepSeek-V3 Embeddings Visualization ({method.upper()}, {len(tokens)} tokens)',
                    xaxis_title=f'{method.upper()} Component 1',
                    yaxis_title=f'{method.upper()} Component 2',
                    hovermode='closest',
                    width=800,
                    height=600
                )
            else:
                fig = go.Figure(data=go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode='markers',
                    text=tokens,
                    hovertemplate='Token: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
                    marker=dict(
                        size=4,
                        opacity=0.7,
                        color=np.arange(len(tokens)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Token Index")
                    )
                ))
                
                fig.update_layout(
                    title=f'DeepSeek-V3 Embeddings Visualization ({method.upper()}, {len(tokens)} tokens)',
                    scene=dict(
                        xaxis_title=f'{method.upper()} Component 1',
                        yaxis_title=f'{method.upper()} Component 2',
                        zaxis_title=f'{method.upper()} Component 3'
                    ),
                    width=800,
                    height=600
                )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            error_fig.update_layout(title="Visualization Error")
            return json.dumps(error_fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')


@app.route('/visualize', methods=['POST'])
def visualize():
    """API endpoint for creating visualizations."""
    data = request.get_json()
    method = data.get('method', 'tsne')
    n_samples = data.get('n_samples', 1000)
    n_components = data.get('n_components', 2)
    
    visualizer = app.config.get('visualizer')
    if not visualizer:
        return jsonify({'error': 'Visualizer not initialized'}), 500
    
    try:
        plot_json = visualizer.create_visualization(method, n_samples, n_components)
        return json.loads(plot_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/info')
def info():
    """Get embedding information."""
    visualizer = app.config.get('visualizer')
    if not visualizer:
        return jsonify({'error': 'Visualizer not initialized'}), 500
    
    try:
        info = visualizer.extractor.get_embedding_info()
        available_methods = visualizer.reducer.get_available_methods()
        info['available_methods'] = available_methods
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
