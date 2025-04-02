# DeepSeek-V3 Attention Visualization Tool

This tool allows you to visualize the attention patterns of the DeepSeek-V3 model on sample user inputs. It provides a web-based interface for comparing attention patterns across different input samples.

## Features

- Extract attention weights from the Multi-head Latent Attention (MLA) mechanism
- Visualize attention patterns as interactive heatmaps
- Compare attention patterns across multiple input samples
- Select specific layers and attention heads to visualize
- Interactive web interface for easy exploration

## Requirements

- Python 3.8+
- PyTorch 2.4.1+
- DeepSeek-V3 model checkpoint
- Other dependencies listed in `requirements.txt`

## Installation

1. Make sure you have the DeepSeek-V3 model checkpoint and configuration file.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the visualization tool using the provided script:

```bash
./run_visualization.sh --ckpt-path /path/to/checkpoint --config /path/to/config.json
```

Optional arguments:
- `--host`: Host to run the server on (default: 127.0.0.1)
- `--port`: Port to run the server on (default: 5000)

## How to Use

1. Open your web browser and navigate to `http://localhost:5000` (or the host/port you specified).
2. Enter one or more text samples in the input fields.
3. Click "Visualize Attention" to process the samples.
4. Use the Layer and Attention Head selectors to explore different parts of the model's attention mechanism.
5. The heatmaps show the attention weights between tokens, with brighter colors indicating stronger attention.
6. Add more samples using the "Add Sample" button to compare attention patterns across different inputs.

## Technical Details

The tool works by modifying the DeepSeek-V3 model to extract attention weights during inference. The attention weights are then processed and visualized using Plotly.js.

The main components are:
- `model.py`: Modified to return attention weights from the MLA mechanism
- `attention_extractor.py`: Utility for extracting attention patterns from the model
- `app.py`: Flask web application for serving the visualization interface
- `templates/index.html`: HTML template for the web interface
- `static/js/visualization.js`: JavaScript for interactive visualization

## Limitations

- Processing large text samples may require significant GPU memory
- Visualization performance may be affected by the number and length of samples
- The tool is designed for exploration and research purposes, not for production use
