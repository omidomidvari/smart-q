// API helper functions
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API Error');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Load model information
async function loadModelInfo() {
    try {
        const info = await apiCall('/api/model-info');
        const infoBox = document.getElementById('model-info');
        infoBox.innerHTML = `
            <p><strong>Model:</strong> ${info.model}</p>
            <p><strong>Device:</strong> ${info.device}</p>
            <p><strong>Messages:</strong> ${info.history_count}</p>
        `;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Send message
async function sendMessage(event) {
    event.preventDefault();
    
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Display user message
    addMessage(message, 'user');
    input.value = '';
    
    // Show loading indicator
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'flex';
    
    try {
        // Send message to server
        const response = await apiCall('/api/chat', 'POST', { message: message });
        
        // Display assistant response
        addMessage(response.assistant, 'assistant');
        
        // Update model info
        loadModelInfo();
    } catch (error) {
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

// Add message to chat
function addMessage(text, sender) {
    const chatMessages = document.getElementById('chat-messages');
    
    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${escapeHtml(text)}
            <span class="message-time">${time}</span>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Escape HTML
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Clear history
async function clearHistory() {
    if (!confirm('Are you sure you want to clear the chat history?')) return;
    
    try {
        await apiCall('/api/clear', 'POST');
        document.getElementById('chat-messages').innerHTML = `
            <div class="welcome-message">
                <h2>👋 Welcome to AI Chatbot</h2>
                <p>Start a conversation with the AI assistant powered by transformer models.</p>
            </div>
        `;
        loadModelInfo();
    } catch (error) {
        alert(`Error clearing history: ${error.message}`);
    }
}

// Update system prompt
async function updateSystemPrompt() {
    const prompt = document.getElementById('system-prompt').value.trim();
    
    if (!prompt) {
        alert('Please enter a system prompt');
        return;
    }
    
    try {
        await apiCall('/api/system-prompt', 'POST', { prompt: prompt });
        alert('System prompt updated!');
    } catch (error) {
        alert(`Error updating system prompt: ${error.message}`);
    }
}

// Save conversation
async function saveConversation() {
    const filename = prompt('Enter filename (without .json):', 'conversation');
    
    if (filename === null) return;
    
    try {
        const response = await apiCall('/api/save', 'POST', { filename: filename });
        alert(`${response.message}`);
    } catch (error) {
        alert(`Error saving conversation: ${error.message}`);
    }
}

// Load conversation history on page load
async function loadHistory() {
    try {
        const data = await apiCall('/api/history');
        const chatMessages = document.getElementById('chat-messages');
        
        if (data.history.length > 0) {
            chatMessages.innerHTML = '';
            data.history.forEach(turn => {
                addMessage(turn.user, 'user');
                addMessage(turn.assistant, 'assistant');
            });
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
    loadHistory();
    
    // Focus on input
    document.getElementById('message-input').focus();
});

// Auto-focus input when page is clicked
document.addEventListener('click', function() {
    document.getElementById('message-input').focus();
});
