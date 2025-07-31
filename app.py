from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import nltk
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import json

app = Flask(__name__)
CORS(app)

# Simple chatbot responses
chat_pairs = [
    [r"hi|hello|hey", ["Hello!", "Hi there!", "How can I help you?"]],
    [r"what is your name", ["I'm an AI assistant!"]],
    [r"how are you", ["I'm just a program, but I'm functioning well!"]],
    [r"(.*) your name (.*)", ["My name is AI Assistant"]],
    [r"(.*) help (.*)", ["I can help with image classification, chat, and recommendations. What would you like to try?"]],
    [r"(.*)", ["I'm not sure I understand. Could you rephrase that?"]]
]

# Recommendation system data
movies = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", "Forrest Gump"]

# Initialize chatbot
chatbot = Chat(chat_pairs, reflections)

# Initialize sentiment analyzer
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')
print("Model loaded successfully!")

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = chatbot.respond(user_input)
    return jsonify({'response': response})

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)
        
        # Preprocess and predict
        processed_image = preprocess_image(temp_path)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        
        # Get top prediction
        _, label, confidence = decoded_predictions[0]
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'prediction': label.replace('_', ' ').title(),
            'confidence': float(confidence),
            'class_name': label.replace('_', ' ').title()
        })
    except Exception as e:
        # Clean up in case of error
        if os.path.exists('temp_image.jpg'):
            os.remove('temp_image.jpg')
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    # In a real app, you would implement a recommendation algorithm here
    recommended = random.sample(movies, min(3, len(movies)))
    return jsonify({'recommendations': recommended})

@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Get sentiment scores
        scores = sia.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return jsonify({
            'sentiment': sentiment,
            'scores': scores,
            'text': text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Download NLTK data
    nltk.download('punkt')
    # Run the app on all available network interfaces
    app.run(host='0.0.0.0', port=5000, debug=True)
