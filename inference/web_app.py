import os
import json
import torch
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
from generate import generate
from pdf_processor import PDFProcessor
from vector_db import VectorDB
from rag_integrator import RAGIntegrator

app = Flask(__name__)
app.secret_key = 'deepseek_rag_system_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

rag_integrator = RAGIntegrator(app.config['UPLOAD_FOLDER'], 'indices')

model = None
tokenizer = None
args = None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_and_tokenizer(ckpt_path, config_path):
    """Load the DeepSeek model and tokenizer"""
    global model, tokenizer, args
    
    with open(config_path) as f:
        args = ModelArgs(**json.load(f))
        
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.device(device):
        model = Transformer(args)
        
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    world_size = 1
    rank = 0
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    
    return True

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload"""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        success = rag_integrator.process_pdf(filepath)
        
        if success:
            flash(f'File {filename} uploaded and processed successfully')
        else:
            flash(f'Error processing file {filename}')
            
        return redirect(url_for('index'))
    else:
        flash('Invalid file type. Only PDF files are allowed')
        return redirect(request.url)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with RAG"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return jsonify({
            "response": "Model not loaded. Please load the model first.",
            "context_used": False
        })
    
    data = request.json
    query = data.get('message', '')
    chat_history = data.get('history', [])
    
    context = rag_integrator.retrieve_context(query)
    
    messages = []
    for message in chat_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    if context:
        system_message = f"You are a helpful assistant. Use the following information from the PDF to answer the user's question. If the information is not in the provided context, say you don't know: \n\n{context}"
        messages.insert(0, {"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": query})
    
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    completion_tokens = generate(model, [prompt_tokens], 500, tokenizer.eos_token_id, 0.7)
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    
    return jsonify({
        "response": completion,
        "context_used": bool(context)
    })

@app.route('/model/load', methods=['POST'])
def load_model_endpoint():
    """Endpoint to load the model and tokenizer"""
    data = request.json
    ckpt_path = data.get('ckpt_path', '')
    config_path = data.get('config_path', '')
    
    if not ckpt_path or not config_path:
        return jsonify({"success": False, "error": "Missing checkpoint or config path"})
    
    try:
        success = load_model_and_tokenizer(ckpt_path, config_path)
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
