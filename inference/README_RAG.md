# DeepSeek-V3 RAG System

This is a simple web-based RAG (Retrieval-Augmented Generation) system that integrates with DeepSeek-V3's existing architecture. It allows you to chat with a PDF document using DeepSeek's powerful language model.

## Features

- Upload and process a single PDF document
- Extract text and create vector embeddings for efficient retrieval
- Retrieve relevant context from the PDF based on user queries
- Generate responses using DeepSeek-V3 model enhanced with retrieved context
- Web interface for easy interaction

## Requirements

All required dependencies are listed in the `requirements.txt` file. The main dependencies include:

- Flask for the web interface
- PyPDF2 for PDF processing
- Sentence-Transformers for creating embeddings
- FAISS for vector storage and retrieval
- DeepSeek-V3 model and its dependencies

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the web server:

```bash
python web_app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Load the model by providing the checkpoint path and config path in the web interface

4. Upload a PDF document

5. Start chatting with the PDF content

### Command Line Interface

You can also use the RAG system from the command line:

```bash
python rag_generate.py --ckpt-path /path/to/checkpoint --config-path /path/to/config.json --pdf-path /path/to/document.pdf --interactive
```

## Architecture

The RAG system consists of the following components:

1. **PDF Processor**: Extracts text from PDF documents and splits it into chunks
2. **Vector Database**: Creates and stores embeddings for text chunks, enables semantic search
3. **RAG Integrator**: Connects PDF processing, vector database, and DeepSeek model
4. **Web Application**: Provides a user interface for uploading PDFs and chatting

## Integration with DeepSeek-V3

The system integrates with DeepSeek-V3's existing architecture by:

1. Using the same model loading approach from `generate.py`
2. Extending the text generation functionality to include retrieved context
3. Maintaining compatibility with DeepSeek's model architecture in `model.py`
4. Using the same chat template format for messages

## Limitations

- Supports only one PDF document at a time
- Performance depends on the quality of the embeddings and retrieval
- Large PDF documents may require significant memory for processing
