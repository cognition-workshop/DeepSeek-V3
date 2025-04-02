import json
import re
import os
from typing import Dict, List, Optional, Any

class TemplateProcessor:
    def __init__(self, templates_path: str, customers_dir: str):
        self.templates_path = templates_path
        self.customers_dir = customers_dir
        self.templates = self._load_templates()
        
    def _load_templates(self) -> List[Dict[str, Any]]:
        """Load templates from JSON file."""
        with open(self.templates_path, 'r') as f:
            data = json.load(f)
        return data.get('templates', [])
        
    def get_customer_data(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Load customer data by ID."""
        customer_path = os.path.join(self.customers_dir, f"{customer_id}.json")
        if not os.path.exists(customer_path):
            return None
            
        with open(customer_path, 'r') as f:
            return json.load(f)
            
    def find_matching_template(self, message: str) -> Optional[Dict[str, Any]]:
        """Find template that matches the input message."""
        message_lower = message.lower()
        
        for template in self.templates:
            for trigger in template.get('triggers', []):
                if trigger.lower() in message_lower:
                    return template
                    
        return None
        
    def _replace_variables(self, template: str, data: Dict[str, Any]) -> str:
        """Replace {{variable}} patterns in template with actual data."""
        pattern = r'{{([^{}]+)}}'
        
        def replace_match(match):
            variable_path = match.group(1).strip()
            
            if variable_path.startswith('#if '):
                condition_var = variable_path[4:].strip()
                value = self._get_nested_value(data, condition_var)
                return str(value) if value else ""
                
            value = self._get_nested_value(data, variable_path)
            return str(value) if value is not None else ""
            
        conditional_pattern = r'{{#if ([^{}]+)}}(.*?){{\/if}}'
        processed = re.sub(conditional_pattern, lambda m: m.group(2) if self._get_nested_value(data, m.group(1).strip()) else "", template, flags=re.DOTALL)
        
        return re.sub(pattern, replace_match, processed)
        
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value from nested dictionaries using dot notation."""
        parts = path.split('.')
        current = data
        
        for part in parts:
            if current is None:
                return None
                
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, dict) and part.endswith('length') and part[:-7] in current:
                return len(current[part[:-7]])
            elif isinstance(current, dict) and hasattr(current, 'get'):
                current = current.get(part)
            else:
                return None
                
        return current
        
    def process_template(self, template_id: str, customer_data: Dict[str, Any], chat_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Process a template with customer data and chat history."""
        template_obj = next((t for t in self.templates if t.get('id') == template_id), None)
        if not template_obj:
            return None
            
        template_text = template_obj.get('template', '')
        
        data = {
            'customer': customer_data,
            'chat_history': {
                'last_topic': chat_history[-1]['topic'] if chat_history and len(chat_history) > 0 else None
            }
        }
        
        return self._replace_variables(template_text, data)
        
    def process_message(self, message: str, customer_id: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Process a message and return a response using templates if applicable."""
        customer_data = self.get_customer_data(customer_id)
        if not customer_data:
            return None
            
        template = self.find_matching_template(message)
        if not template:
            return None
            
        return self.process_template(template['id'], customer_data, chat_history)
