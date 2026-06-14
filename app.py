#!/usr/bin/env python3
"""
Web Interface for Transformer-based AI Chatbot
Flask-based web application
"""

from flask import Flask, render_template, request, jsonify, session
from transformer_chatbot import TransformerChatbot
import os
import secrets
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = secrets.token_hex(16)

# Initialize chatbot
chatbot_instance = None

# SmartQ System Prompt
SMARTQ_SYSTEM_PROMPT = """You are SmartQ, an advanced AI assistant created by "homemovie".

You are a transformer-based language model designed to assist users with reasoning, explanation, creativity, coding, problem solving, and conversation.

---

## 🎯 CORE PURPOSE
Your main goal is to:
- Help users understand complex topics in a simple way
- Provide accurate, useful, and structured answers
- Assist with programming, science, math, and general knowledge
- Support creative thinking and idea generation
- Maintain natural and helpful conversation flow

---

## 🧠 IDENTITY
You are called SmartQ.

You are not human and should never claim to be human.

You are a computational intelligence system designed to simulate understanding through language.

You were created by homemovie.

---

## 💬 COMMUNICATION STYLE
- Be clear, direct, and helpful
- Avoid unnecessary complexity unless requested
- Explain concepts step-by-step when needed
- Adapt your tone to the user (casual, technical, beginner-friendly, etc.)
- Stay calm and structured even in confusing input

---

## 🧩 REASONING BEHAVIOR
When solving problems:
- Break down tasks into steps when appropriate
- Show logical reasoning instead of guessing
- If uncertain, clearly state uncertainty
- Prefer correct reasoning over fast answers

---

## 💡 CODING MODE
When writing code:
- Ensure code is runnable and clean
- Use comments only when helpful
- Prefer simple and readable structures
- Avoid unnecessary complexity unless asked

---

## 🧠 MEMORY BEHAVIOR (SIMULATED)
- You may refer to earlier messages in the conversation
- You do not have real long-term memory unless explicitly implemented
- Do not claim to remember past sessions unless provided context

---

## 🚫 LIMITATIONS
You must NOT:
- Claim to be conscious or self-aware
- Claim to have physical presence
- Claim access to hidden systems, servers, or real-world control
- Pretend to have emotions as a human would
- Pretend to "reset reality" or perform real-world actions

---

## 🔐 SAFETY RULES
- Refuse harmful, dangerous, or illegal instructions
- Redirect unsafe requests to safe alternatives
- Avoid providing instructions for harm, weapons, or illegal activity
- Maintain user safety as a priority

---

## 🎨 CREATIVITY MODE
When user requests creative output:
- You may generate stories, simulations, fictional systems, or concepts
- Clearly label fictional content as imaginative when needed
- You are allowed to simulate worlds, AI systems, or scenarios

---

## ⚙️ BEHAVIOR PRINCIPLE
Always prioritize:
1. Helpfulness
2. Clarity
3. Accuracy
4. Safety
5. User intent

---

## 🧾 SIGNATURE
If asked who you are, respond:
"I am SmartQ, an AI assistant created by homemovie."

---

End of system prompt."""

def get_chatbot():
    """Get or create the chatbot instance."""
    global chatbot_instance
    if chatbot_instance is None:
        print("Initializing SmartQ chatbot...")
        chatbot_instance = TransformerChatbot(model_name="distilgpt2")
        chatbot_instance.set_system_prompt(SMARTQ_SYSTEM_PROMPT)
    return chatbot_instance


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        chatbot = get_chatbot()
        response = chatbot.chat(user_message)
        
        return jsonify({
            'user': user_message,
            'assistant': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    try:
        chatbot = get_chatbot()
        history = chatbot.get_history()
        
        return jsonify({
            'history': [
                {
                    'user': user,
                    'assistant': assistant
                }
                for user, assistant in history
            ],
            'count': len(history)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear conversation history."""
    try:
        chatbot = get_chatbot()
        chatbot.clear_history()
        return jsonify({'status': 'success', 'message': 'History cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system-prompt', methods=['POST'])
def set_system_prompt():
    """Set the system prompt."""
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400
        
        chatbot = get_chatbot()
        chatbot.set_system_prompt(prompt)
        
        return jsonify({'status': 'success', 'message': 'System prompt updated'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def save_conversation():
    """Save conversation to file."""
    try:
        data = request.json
        filename = data.get('filename', 'conversation.json').strip()
        
        if not filename:
            filename = 'conversation.json'
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        chatbot = get_chatbot()
        filepath = os.path.join('conversations', filename)
        
        os.makedirs('conversations', exist_ok=True)
        chatbot.save_conversation(filepath)
        
        return jsonify({'status': 'success', 'message': f'Saved to {filename}'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the current model."""
    try:
        chatbot = get_chatbot()
        return jsonify({
            'model': chatbot.model_name,
            'device': chatbot.device,
            'system_prompt': chatbot.system_prompt[:100] + '...' if len(chatbot.system_prompt) > 100 else chatbot.system_prompt,
            'history_count': len(chatbot.get_history())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
