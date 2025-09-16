# DeepSeek-V3 Embedding Visualizer

This tool provides interactive visualization of DeepSeek-V3 embeddings using dimensionality reduction techniques.

## Overview

The DeepSeek-V3 Embedding Visualizer extracts embeddings from the ParallelEmbedding layer and creates interactive 2D/3D visualizations using dimensionality reduction methods like t-SNE, UMAP, and PCA.

## Features

- **Embedding Extraction**: Extracts embeddings from the ParallelEmbedding layer, handling distributed weights correctly
- **Dimensionality Reduction**: Supports t-SNE, UMAP, and PCA for reducing high-dimensional embeddings
- **Interactive Visualization**: Web-based interface with hover information and zoom/pan capabilities
- **Distributed Support**: Handles distributed embedding weights across multiple devices
- **Configurable Parameters**: Adjustable sample sizes, reduction methods, and visualization dimensions

## Installation

1. Install the base DeepSeek-V3 requirements:
```bash
cd inference
pip install -r requirements.txt
```

2. Or install visualization-specific requirements separately:
```bash
pip install -r visualization/requirements.txt
```

## Usage

### Basic Usage

```bash
python visualize_embeddings.py --ckpt-path /path/to/model --config configs/config_16B.json
```

### Advanced Usage

```bash
python visualize_embeddings.py \
    --ckpt-path /path/to/model \
    --config configs/config_16B.json \
    --port 8080 \
    --host 0.0.0.0 \
    --debug
```

### Parameters

- `--ckpt-path`: Path to the model checkpoint directory (required)
- `--config`: Path to model configuration file (required)
- `--port`: Port for web interface (default: 5000)
- `--host`: Host for web interface (default: localhost)
- `--debug`: Enable debug mode for development

## Web Interface

Once started, open your browser to the displayed URL (e.g., http://localhost:5000) to access the interactive visualization interface.

### Interface Features

- **Reduction Method**: Choose between t-SNE, UMAP, or PCA
- **Number of Samples**: Control how many embeddings to visualize (100-10,000)
- **Dimensions**: Switch between 2D and 3D visualizations
- **Model Info**: View embedding layer information and available methods

### Visualization Methods

1. **t-SNE**: Good for revealing local structure and clusters
2. **UMAP**: Preserves both local and global structure
3. **PCA**: Linear method, fastest but may miss non-linear patterns

## Model Configurations

The tool supports different DeepSeek-V3 model configurations:

- **16B Model** (`config_16B.json`): vocab_size=102,400, dim=2,048
- **236B Model** (`config_236B.json`): vocab_size=102,400, dim=5,120  
- **671B Model** (`config_671B.json`): vocab_size=129,280, dim=7,168

## Technical Details

### Embedding Extraction

The `EmbeddingExtractor` class handles:
- Distributed weight aggregation across devices
- Token ID validation and filtering
- Conversion to numpy arrays for processing

### Dimensionality Reduction

The `DimensionalityReducer` class provides:
- Configurable parameters for each method
- Automatic parameter adjustment for small datasets
- Model persistence for analysis

### Web Application

The Flask-based web application offers:
- Real-time visualization updates
- Interactive Plotly charts
- Error handling and user feedback
- Model information display

## Troubleshooting

### Common Issues

1. **UMAP not available**: Install with `pip install umap-learn`
2. **CUDA out of memory**: Reduce the number of samples
3. **Port already in use**: Change the port with `--port` parameter
4. **Model loading errors**: Verify checkpoint path and config file

### Performance Tips

- Start with smaller sample sizes (500-1000) for faster visualization
- Use PCA for quick exploration, t-SNE/UMAP for detailed analysis
- For large vocabularies, consider sampling specific token ranges

## Development

### Project Structure

```
visualization/
├── __init__.py
├── embedding_extractor.py    # Embedding extraction logic
├── dimensionality_reduction.py # t-SNE, UMAP, PCA implementations
├── web_app.py                # Flask application
├── templates/
│   └── index.html           # Web interface
├── requirements.txt         # Dependencies
└── README.md               # This file
```

### Extending the Tool

To add new visualization methods:
1. Extend `DimensionalityReducer` class
2. Update the web interface dropdown
3. Add method-specific parameters as needed

## License

This tool is part of the DeepSeek-V3 project and follows the same licensing terms.
