import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

from template_processor import TemplateProcessor
from model_integration import ModelWrapper

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)

TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), 'data/templates.json')
CUSTOMERS_DIR = os.path.join(os.path.dirname(__file__), 'data/customers')
MODEL_CKPT_PATH = os.environ.get('MODEL_CKPT_PATH', '/path/to/model/checkpoint')
MODEL_CONFIG_PATH = os.environ.get('MODEL_CONFIG_PATH', '/path/to/model/config.json')

template_processor = TemplateProcessor(TEMPLATES_PATH, CUSTOMERS_DIR)
model_wrapper = None  # Lazy initialization to avoid loading model during app startup

chat_histories = {}

class MessageRequest(BaseModel):
    customer_id: str
    message: str
    use_templates: bool = True

class Message(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatHistory(BaseModel):
    customer_id: str
    messages: List[Message] = []
    
@app.route('/')
def index():
    """Render the web app frontend."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate responses."""
    data = request.json
    
    try:
        req = MessageRequest(**data)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
    if req.customer_id not in chat_histories:
        chat_histories[req.customer_id] = ChatHistory(customer_id=req.customer_id)
        
    history = chat_histories[req.customer_id]
    
    user_message = Message(role="user", content=req.message)
    history.messages.append(user_message)
    
    response_content = None
    if req.use_templates:
        chat_history_context = []
        for msg in history.messages[-10:]:  # Use last 10 messages for context
            if msg.role == "user":
                chat_history_context.append({
                    "timestamp": msg.timestamp,
                    "topic": msg.content[:50],  # Simplified topic extraction
                    "sentiment": "neutral"
                })
                
        response_content = template_processor.process_message(
            req.message, 
            req.customer_id, 
            chat_history_context
        )
        
    if response_content is None:
        global model_wrapper
        if model_wrapper is None:
            model_wrapper = ModelWrapper(MODEL_CKPT_PATH, MODEL_CONFIG_PATH)
            model_wrapper.initialize()
            
        messages = [{"role": m.role, "content": m.content} for m in history.messages]
        
        response_content = model_wrapper.generate_response(messages)
        
    assistant_message = Message(role="assistant", content=response_content)
    history.messages.append(assistant_message)
    
    return jsonify({
        "response": response_content,
        "used_template": response_content != None and req.use_templates
    })

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get all available templates."""
    with open(TEMPLATES_PATH, 'r') as f:
        templates = json.load(f)
    return jsonify(templates)

@app.route('/api/templates', methods=['POST'])
def add_template():
    """Add a new template."""
    template_data = request.json
    
    with open(TEMPLATES_PATH, 'r') as f:
        templates = json.load(f)
        
    templates['templates'].append(template_data)
    
    with open(TEMPLATES_PATH, 'w') as f:
        json.dump(templates, f, indent=2)
        
    template_processor.templates = template_processor._load_templates()
    
    return jsonify({"status": "success"})

@app.route('/api/customers/<customer_id>', methods=['GET'])
def get_customer(customer_id):
    """Get customer data by ID."""
    customer_data = template_processor.get_customer_data(customer_id)
    if customer_data:
        return jsonify(customer_data)
    return jsonify({"error": "Customer not found"}), 404

@app.route('/api/customers/<customer_id>/history', methods=['GET'])
def get_chat_history(customer_id):
    """Get chat history for a customer."""
    if customer_id in chat_histories:
        return jsonify(chat_histories[customer_id].dict())
    return jsonify({"customer_id": customer_id, "messages": []})

if __name__ == '__main__':
    os.makedirs(CUSTOMERS_DIR, exist_ok=True)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
