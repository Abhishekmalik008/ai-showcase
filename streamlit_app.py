import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import nltk
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page config
st.set_page_config(
    page_title="AI Showcase",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize chatbot
chat_pairs = [
    [r"hi|hello|hey", ["Hello!", "Hi there!", "How can I help you?"]],
    [r"what is your name", ["I'm an AI assistant!"]],
    [r"how are you", ["I'm just a program, but I'm functioning well!"]]
]
chatbot = Chat(chat_pairs, reflections)

# Load the MobileNetV2 model
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Sidebar
st.sidebar.title("AI Showcase")
app_mode = st.sidebar.selectbox("Choose an AI feature:", ["Chat with AI", "Image Classification"])

# Main content
st.title("🤖 AI Showcase")

if app_mode == "Chat with AI":
    st.header("💬 Chat with AI")
    
    # Display chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.write(message)
    
    # Chat input
    user_input = st.chat_input("Type a message...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append(("user", user_input))
        
        # Get bot response
        response = chatbot.respond(user_input.lower())
        
        # If no response from patterns, analyze sentiment
        if not response:
            sentiment = sia.polarity_scores(user_input)
            if sentiment['compound'] > 0.2:
                response = "That sounds positive! 😊"
            elif sentiment['compound'] < -0.2:
                response = "I'm sorry to hear that. 😔"
            else:
                response = "Interesting. Tell me more."
        
        # Add bot response to chat
        st.session_state.chat_history.append(("assistant", response))
        
        # Rerun to update the chat display
        st.rerun()

else:  # Image Classification
    st.header("🖼️ Image Classification")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_column_width=True)
        
        # Classify button
        if st.button("Classify Image"):
            with st.spinner("Classifying image..."):
                try:
                    # Preprocess and predict
                    img_array = preprocess_image(image_display)
                    predictions = model.predict(img_array)
                    decoded_predictions = decode_predictions(predictions, top=3)[0]
                    
                    # Display results
                    st.subheader("Predictions:")
                    for i, (_, label, confidence) in enumerate(decoded_predictions):
                        st.write(f"{i+1}. {label.replace('_', ' ').title()} ({(confidence * 100):.2f}%)")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Add some styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
    }
    </style>
""", unsafe_allow_html=True)
