from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import random
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Simple responses for testing
responses = {
    "hello": ["Hi there!", "Hello!", "Hey! How can I help?"],
    "how are you": ["I'm just a computer program, but I'm functioning well!", "I'm doing great, thanks for asking!"],
    "bye": ["Goodbye!", "See you later!", "Have a great day!"]
}

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple AI Chat</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .chat-container {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            #chat-messages {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 10px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                max-width: 70%;
            }
            .user-message {
                background-color: #e3f2fd;
                margin-left: 30%;
            }
            .bot-message {
                background-color: #f1f1f1;
                margin-right: 30%;
            }
            #user-input {
                width: 80%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px 0 0 5px;
            }
            button {
                width: 18%;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 0 5px 5px 0;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>Simple AI Chat</h1>
            <div id="chat-messages"></div>
            <div>
                <input type="text" id="user-input" placeholder="Type your message..." onkeypress="if(event.key === 'Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            function addMessage(message, isUser) {
                const chatMessages = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                messageDiv.textContent = message;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            async function sendMessage() {
                const input = document.getElementById('user-input');
                const message = input.value.trim();
                
                if (!message) return;
                
                // Add user message
                addMessage(message, true);
                input.value = '';
                
                try {
                    // Simple response logic
                    const lowerMsg = message.toLowerCase();
                    let response = "I'm a simple chat bot. Try saying 'hello' or 'how are you?'";
                    
                    if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('hey')) {
                        response = randomResponse("hello");
                    } else if (lowerMsg.includes('how are')) {
                        response = randomResponse("how are you");
                    } else if (lowerMsg.includes('bye') || lowerMsg.includes('goodbye')) {
                        response = randomResponse("bye");
                    } else if (lowerMsg.includes('time') || lowerMsg.includes('date')) {
                        response = 'The current date and time is: ' + new Date().toLocaleString();
                    }
                    
                    // Simulate typing delay
                    await new Promise(resolve => setTimeout(resolve, 500));
                    addMessage(response, false);
                    
                } catch (error) {
                    console.error('Error:', error);
                    addMessage("Sorry, I encountered an error. Please try again.", false);
                }
            }
            
            function randomResponse(key) {
                const possibleResponses = responses[key] || ["I'm not sure how to respond to that."];
                return possibleResponses[Math.floor(Math.random() * possibleResponses.length)];
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    port = 5003
    print(f"\nSimple Chat Server is running on http://localhost:{port}")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=port)
