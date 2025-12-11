# DeepSeek-V3 Embedding Visualizer

A locally hosted web application that visualizes embeddings extracted from PDF files using dimensionality reduction techniques (PCA and t-SNE).

## Overview

This application provides an interactive way to visualize text embeddings from PDF documents. It uses a simplified embedding approach inspired by DeepSeek-V3's `ParallelEmbedding` class (defined in `inference/model.py` lines 87-126) with the same embedding dimension of 2048.

## Features

- **PDF Upload**: Drag and drop or browse to upload PDF files
- **Text Extraction**: Automatically extracts and chunks text from PDFs
- **Embedding Generation**: Generates 2048-dimensional embeddings for each text chunk
- **Dimensionality Reduction**: Supports both PCA and t-SNE for reducing embeddings to 2D
- **Interactive Visualization**: Scatter plot visualization with hover tooltips and click-to-select functionality
- **Configurable Parameters**: Adjust chunk size and reduction method

## Architecture

### Backend (FastAPI)

The backend provides REST API endpoints for:
- `/upload-pdf` - Upload and extract text from PDF files
- `/generate-embeddings` - Generate embeddings and perform dimensionality reduction
- `/model-info` - Get information about the embedding model configuration

The embedding implementation is inspired by DeepSeek-V3's architecture:
- Vocabulary size: 102,400 (matching `ModelArgs.vocab_size`)
- Embedding dimension: 2,048 (matching `ModelArgs.dim`)
- Reference: `inference/model.py` lines 87-126

### Frontend (React + TypeScript)

Built with:
- React 18 with TypeScript
- Vite for fast development
- Tailwind CSS for styling
- Recharts for interactive scatter plot visualization
- Lucide React for icons

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Installation

1. **Install backend dependencies:**
   ```bash
   cd webapp/backend
   pip install -r requirements.txt
   ```

2. **Install frontend dependencies:**
   ```bash
   cd webapp/frontend
   npm install
   ```

### Running the Application

1. **Start the backend server:**
   ```bash
   cd webapp/backend
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend development server:**
   ```bash
   cd webapp/frontend
   npm run dev
   ```

3. **Open your browser** and navigate to `http://localhost:5173`

## Usage

1. **Upload a PDF**: Drag and drop a PDF file or click to browse
2. **Configure**: Adjust chunk size and select reduction method (PCA or t-SNE)
3. **Visualize**: Click "Generate & Visualize" to see the embedding visualization
4. **Explore**: Hover over points to see text previews, click to view full content

## API Reference

### POST /upload-pdf

Upload a PDF file and extract text chunks.

**Query Parameters:**
- `chunk_size` (int, default: 500): Size of each text chunk in characters
- `overlap` (int, default: 50): Overlap between consecutive chunks

**Response:**
```json
{
  "text_chunks": ["chunk1", "chunk2", ...],
  "total_pages": 10,
  "total_characters": 5000
}
```

### POST /generate-embeddings

Generate embeddings and reduce dimensionality.

**Request Body:**
```json
{
  "text_chunks": ["text1", "text2", ...],
  "reduction_method": "pca",  // or "tsne"
  "n_components": 2  // 2 or 3
}
```

**Response:**
```json
{
  "reduced_embeddings": [[x1, y1], [x2, y2], ...],
  "original_dim": 2048,
  "n_chunks": 10,
  "text_chunks": ["text1", "text2", ...],
  "reduction_method": "pca"
}
```

## Technical Details

### Embedding Generation

The `SimpleEmbedding` class provides a simplified embedding layer that:
1. Tokenizes text using character-level encoding
2. Maps tokens to 2048-dimensional vectors
3. Applies mean pooling to get a single embedding per chunk

This approach maintains compatibility with DeepSeek-V3's embedding dimension while being lightweight enough to run locally without GPU requirements.

### Dimensionality Reduction

- **PCA (Principal Component Analysis)**: Fast, deterministic, preserves global structure
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Better at preserving local structure, reveals clusters

## License

This project is part of the DeepSeek-V3 repository. See the main LICENSE files for details.
