from flask import Flask, render_template, jsonify, request, session, send_from_directory
from flask_cors import CORS
import nltk
from nltk.chat.util import Chat, reflections
import random
import json
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import spacy
from dateutil import parser
import wikipediaapi
import time
import uuid
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker for more creative results
    ).to(device)
    # Enable attention slicing for lower memory usage
    pipe.enable_attention_slicing()
    print("Stable Diffusion model loaded successfully!")
except Exception as e:
    print(f"Error loading Stable Diffusion model: {e}")
    pipe = None

# Configure upload folder for generated images
UPLOAD_FOLDER = 'static/generated_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent='AIShowcaseBot/1.0 (your-email@example.com)'
)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize conversation memory and context
conversation_memory = {}
conversation_context = {
    'current_topic': None,
    'previous_topics': [],
    'sentiment': 'neutral',
    'entities': {},
    'intent': None,
    'needs_clarification': False,
    'pending_action': None
}

# Load knowledge base
def load_knowledge_base():
    knowledge_file = 'knowledge_base.json'
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r') as f:
            return json.load(f)
    return {
        'users': {},
        'facts': {},
        'preferences': {}
    }

# Save knowledge base
def save_knowledge_base(knowledge_base):
    with open('knowledge_base.json', 'w') as f:
        json.dump(knowledge_base, f, indent=2)

# Remember user information
def remember_user_info(user_id, info_type, info):
    knowledge_base = load_knowledge_base()
    if user_id not in knowledge_base['users']:
        knowledge_base['users'][user_id] = {}
    knowledge_base['users'][user_id][info_type] = info
    knowledge_base['users'][user_id]['last_seen'] = str(datetime.now())
    save_knowledge_base(knowledge_base)

# Get user information
def get_user_info(user_id, info_type):
    knowledge_base = load_knowledge_base()
    return knowledge_base['users'].get(user_id, {}).get(info_type)

# Add a new fact to the knowledge base
def add_fact(fact_type, fact_data):
    knowledge_base = load_knowledge_base()
    if fact_type not in knowledge_base['facts']:
        knowledge_base['facts'][fact_type] = []
    knowledge_base['facts'][fact_type].append({
        'data': fact_data,
        'timestamp': str(datetime.now())
    })
    save_knowledge_base(knowledge_base)

# Get relevant facts from knowledge base
def get_relevant_facts(query, fact_type=None):
    knowledge_base = load_knowledge_base()
    relevant_facts = []
    
    if fact_type:
        fact_types = [fact_type] if fact_type in knowledge_base['facts'] else []
    else:
        fact_types = knowledge_base['facts'].keys()
    
    for ft in fact_types:
        for fact in knowledge_base['facts'].get(ft, []):
            if any(word.lower() in fact['data'].lower() for word in query.split()):
                relevant_facts.append(fact['data'])
    
    return relevant_facts

# Helper functions for mathematical operations
def calculate_expression(match):
    try:
        num1 = int(match.group(2))
        operator = match.group(3)
        num2 = int(match.group(4))
        
        if operator == '+':
            return f"{num1} + {num2} = {num1 + num2}"
        elif operator == '-':
            return f"{num1} - {num2} = {num1 - num2}"
        elif operator == '*':
            return f"{num1} * {num2} = {num1 * num2}"
        elif operator == '/':
            if num2 == 0:
                return "Error: Division by zero is not allowed."
            return f"{num1} / {num2} = {num1 / num2:.2f}"
    except Exception as e:
        return f"I couldn't process that calculation. Please try again with a valid expression."

def calculate_math_operation(match):
    try:
        operation = match.group(1).lower()
        num = float(match.group(2))
        
        if operation == 'square':
            return f"The square of {num} is {num ** 2}"
        elif operation == 'square root':
            if num < 0:
                return "Cannot calculate square root of a negative number."
            return f"The square root of {num} is {math.sqrt(num):.4f}"
        elif operation == 'cube':
            return f"The cube of {num} is {num ** 3}"
        elif operation == 'cube root':
            return f"The cube root of {num} is {num ** (1/3):.4f}"
        elif operation == 'factorial':
            if num < 0:
                return "Factorial is not defined for negative numbers."
            if num > 20:  # To prevent very large numbers
                return "That number is too large for me to calculate the factorial."
            return f"The factorial of {int(num)} is {math.factorial(int(num))}"
    except Exception as e:
        return "I couldn't process that mathematical operation. Please try again."

def check_prime(match):
    try:
        # Extract the number from the match
        num = int(''.join(filter(str.isdigit, match.string)))
        
        if num < 2:
            return f"{num} is not a prime number."
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                return f"{num} is not a prime number. It's divisible by {i}."
        return f"Yes, {num} is a prime number!"
    except:
        return "I couldn't check if that's a prime number. Please provide a valid number."

# Helper functions for science and technology
def get_science_fact(match):
    science_facts = [
        "The human body contains enough DNA to stretch from the Sun to Pluto and back 17 times.",
        "A single bolt of lightning contains enough energy to toast 100,000 slices of bread.",
        "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat.",
        "The average cloud weighs about 1.1 million pounds - that's about the same as 100 elephants!",
        "There are more possible iterations of a game of chess than there are atoms in the known universe.",
        "A day on Venus is longer than a year on Venus - it takes 243 Earth days to rotate on its axis but only 225 Earth days to orbit the Sun.",
        "The human brain can store an estimated 2.5 petabytes of information - that's about 3 million hours of TV shows.",
        "A teaspoonful of neutron star material would weigh about 6 billion tons - that's about 900 Great Pyramids of Giza!"
    ]
    return random.choice(science_facts)

def get_tech_fact(match):
    tech_facts = [
        "The first computer virus was created in 1983 by Fred Cohen as an experiment.",
        "The first 1GB hard drive, introduced in 1980, weighed about 550 pounds and cost $40,000.",
        "The first computer mouse was made of wood and was created in 1964 by Douglas Engelbart.",
        "The world's first webcam was created at the University of Cambridge to monitor a coffee pot.",
        "The first computer programmer was a woman named Ada Lovelace, who wrote the first algorithm in the 1840s.",
        "The first website went live in 1991 and is still online at http://info.cern.ch.",
        "The average person spends about 6 years and 8 months of their life on social media.",
        "The first electronic computer ENIAC weighed more than 27 tons and consumed 150 kilowatts of power."
    ]
    return random.choice(tech_facts)

def get_space_fact(match):
    space_facts = [
        "A day on Mercury lasts 1,408 hours - that's 58.7 Earth days!",
        "The footprints on the Moon will stay there for millions of years because there's no wind to blow them away.",
        "One million Earths could fit inside the Sun.",
        "The sunset on Mars appears blue because of the way the dust in the atmosphere scatters light.",
        "A year on Venus is shorter than a day on Venus.",
        "The International Space Station travels at about 17,500 mph - that's about 5 miles per second!",
        "If two pieces of the same type of metal touch in space, they will permanently bond together in a process called cold welding.",
        "There's a planet made of diamonds called 55 Cancri e that's twice the size of Earth."
    ]
    return random.choice(space_facts)

# Helper functions for general knowledge
def get_leader_info(match):
    leaders = {
        'usa': {'title': 'President', 'name': 'Joe Biden'},
        'uk': {'title': 'Prime Minister', 'name': 'Rishi Sunak'},
        'india': {'title': 'Prime Minister', 'name': 'Narendra Modi'},
        'australia': {'title': 'Prime Minister', 'name': 'Anthony Albanese'},
        'canada': {'title': 'Prime Minister', 'name': 'Justin Trudeau'},
        'japan': {'title': 'Prime Minister', 'name': 'Fumio Kishida'},
        'china': {'title': 'President', 'name': 'Xi Jinping'},
        'russia': {'title': 'President', 'name': 'Vladimir Putin'},
        'germany': {'title': 'Chancellor', 'name': 'Olaf Scholz'},
        'france': {'title': 'President', 'name': 'Emmanuel Macron'}
    }
    
    country = match.group(3).lower()
    if country in leaders:
        leader = leaders[country]
        return f"The current {leader['title']} of {country.title()} is {leader['name']}."
    return f"I don't have information about the leader of {country.title()}."

def get_capital(match):
    capitals = {
        'france': 'Paris',
        'japan': 'Tokyo',
        'germany': 'Berlin',
        'italy': 'Rome',
        'spain': 'Madrid',
        'china': 'Beijing',
        'russia': 'Moscow',
        'uk': 'London',
        'usa': 'Washington, D.C.',
        'canada': 'Ottawa',
        'australia': 'Canberra',
        'india': 'New Delhi',
        'brazil': 'Brasília',
        'mexico': 'Mexico City',
        'south africa': 'Pretoria',
        'egypt': 'Cairo',
        'thailand': 'Bangkok',
        'south korea': 'Seoul',
        'indonesia': 'Jakarta'
    }
    
    country = match.group(2).lower()
    if country in capitals:
        return f"The capital of {country.title()} is {capitals[country]}."
    return f"I don't know the capital of {country.title()}. Could you try another country?"

def get_population(match):
    populations = {
        'china': '1.4 billion',
        'india': '1.4 billion',
        'usa': '331 million',
        'indonesia': '276 million',
        'pakistan': '225 million',
        'brazil': '213 million',
        'nigeria': '211 million',
        'bangladesh': '166 million',
        'russia': '146 million',
        'mexico': '130 million',
        'japan': '126 million',
        'ethiopia': '117 million',
        'philippines': '111 million',
        'egypt': '104 million',
        'vietnam': '98 million',
        'dr congo': '90 million',
        'turkey': '85 million',
        'iran': '85 million',
        'germany': '83 million',
        'thailand': '70 million',
        'uk': '67 million',
        'france': '65 million',
        'italy': '60 million',
        'south africa': '60 million',
        'south korea': '51 million',
        'spain': '47 million',
        'argentina': '45 million',
        'algeria': '44 million',
        'canada': '38 million',
        'australia': '26 million',
        'taiwan': '23.6 million',
        'sri lanka': '21.5 million',
        'netherlands': '17.5 million',
        'chile': '19.5 million',
        'sweden': '10.4 million',
        'switzerland': '8.7 million',
        'singapore': '5.7 million',
        'new zealand': '5.1 million',
        'ireland': '5.0 million',
        'costa rica': '5.1 million',
        'norway': '5.4 million',
        'denmark': '5.8 million',
        'finland': '5.5 million',
        'qatar': '2.9 million',
        'kuwait': '4.3 million',
        'uae': '9.9 million',
        'israel': '9.3 million'
    }
    
    country = match.group(2).lower()
    if country in populations:
        return f"The population of {country.title()} is approximately {populations[country]}."
    return f"I don't have the population data for {country.title()}. Could you try another country?"

def get_historical_figure(match):
    figures = {
        'mahatma gandhi': "Mahatma Gandhi (1869-1948) was an Indian lawyer and anti-colonial nationalist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule.",
        'albert einstein': "Albert Einstein (1879-1955) was a German-born theoretical physicist, best known for developing the theory of relativity, one of the two pillars of modern physics (alongside quantum mechanics).",
        'isaac newton': "Sir Isaac Newton (1643-1727) was an English mathematician, physicist, and astronomer who is widely recognized as one of the most influential scientists of all time and a key figure in the scientific revolution.",
        'marie curie': "Marie Curie (1867-1934) was a Polish-born physicist and chemist who conducted pioneering research on radioactivity, the first woman to win a Nobel Prize, and the only person to win Nobel Prizes in two different scientific fields.",
        'leonardo da vinci': "Leonardo da Vinci (1452-1519) was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect.",
        'nikola tesla': "Nikola Tesla (1856-1943) was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system."
    }
    
    figure = match.group(2).lower()
    if figure in figures:
        return figures[figure]
    return f"I don't have detailed information about {figure.title()}. Could you ask about someone else?"

def get_historical_event(match):
    events = {
        'world war i': "World War I took place from 1914 to 1918. It was a global war originating in Europe that drew in all the world's economic great powers, assembled in two opposing alliances: the Allies and the Central Powers.",
        'world war ii': "World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all of the great powers—forming two opposing military alliances: the Allies and the Axis.",
        'the american revolution': "The American Revolution was an ideological and political revolution that occurred in colonial North America between 1765 and 1783. The American Revolution led to the creation of the United States of America.",
        'the french revolution': "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended in November 1799 with the formation of the French Consulate.",
        'the industrial revolution': "The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States, in the period from about 1760 to sometime between 1820 and 1840."
    }
    
    event = match.group(2).lower()
    if event in events:
        return events[event]
    return f"I don't have detailed information about {event}. Could you ask about another historical event?"

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize knowledge base if it doesn't exist
if not os.path.exists('knowledge_base.json'):
    save_knowledge_base({
        'users': {},
        'facts': {},
        'preferences': {}
    })

# Enhanced chatbot responses and capabilities
chat_pairs = [
    # Greetings
    [r"hi|hello|hey|greetings|salutations", 
     ["Hello! I'm your AI assistant. How can I help you today?", 
      "Hi there! I'm here to chat and answer your questions. What's on your mind?",
      "Hey! I'm your AI companion. What would you like to talk about?",
      "Greetings! I'm here to assist you. How can I help you today?"]],
    
    # Name
    [r"what('?s| is) your name\??", 
     ["I'm an AI assistant created to help and chat with you!", 
      "You can call me ChatBot. I'm your virtual assistant.", 
      "I'm your friendly AI assistant! I don't have a name, but you can call me whatever you like."]],
    
    # How are you
    [r"how (are you|do you feel|'re you)", 
     ["I'm just a program, but I'm functioning perfectly! How about you?", 
      "I don't have feelings, but I'm here and ready to help you with anything you need!", 
      "All systems go! I'm ready to assist you. How can I help you today?"]],
    
    # Thank you
    [r"thank|thanks|thx", 
     ["You're welcome! Is there anything else I can help you with?", 
      "Happy to help! Let me know if you need anything else.", 
      "Anytime! That's what I'm here for.", 
      "No problem at all! Feel free to ask me anything else."]],
    
    # Help
    [r"help|what can you do", 
     ["I can chat with you, answer questions, recommend movies, and more! Try asking me about technology, science, or just say 'tell me a fact'.", 
      "I'm here to chat, answer questions, and recommend movies. You can ask me about various topics like technology, science, movies, or just have a casual conversation!"]],
    
    # Time
    [r"what('?s the)? time\??", 
     [f"I don't have access to real-time data, but you can check your device's clock."]],
    
    # Weather
    [r"(what'?s the )?weather (like|today|now)", 
     ["I don't have access to weather data, but you can check a weather app or website!"]],
    
    # Jokes and Humor
    [r"(tell me a|do you know a|say a|know any) (joke|jokes|funny)", 
     ["Why don't scientists trust atoms? Because they make up everything!",
      "Did you hear about the mathematician who's afraid of negative numbers? He'll stop at nothing to avoid them!",
      "Why did the computer go to the doctor? It had a virus!",
      "What do you call a fake noodle? An impasta!"
     ]],
     
    # Technology
    [r"(what is|tell me about|explain) (ai|artificial intelligence|machine learning|chatbot|chat bot)",
     ["Artificial Intelligence (AI) is the simulation of human intelligence in machines, enabling them to perform tasks that typically require human intelligence.",
      "Machine learning is a subset of AI that allows systems to learn and improve from experience without being explicitly programmed.",
      "A chatbot is a software application designed to simulate human conversation. I'm an example of a chatbot that can understand and respond to natural language inputs!"
     ]],
     
    # Science
    [r"(tell me a|do you know a|share a) (science|scientific|space|physics|biology|chemistry) (fact|facts)",
     ["A day on Venus is longer than a year on Venus! It takes Venus 243 Earth days to rotate once on its axis.",
      "The human brain contains about 86 billion neurons, each connected to up to 10,000 other neurons.",
      "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old!"
     ]],
     
    # Movies
    [r"(recommend|suggest|what is|tell me about) (a|some) (good|great|best) (movie|film|movies|films)",
     ["I recommend 'Inception' - it's a mind-bending science fiction film about dreams within dreams.",
      "If you enjoy thought-provoking movies, you might like 'The Matrix' - it explores themes of reality and AI.",
      "For something lighter, 'The Grand Budapest Hotel' is a visually stunning and quirky comedy-drama."
     ]],
     
    # Fun facts
    [r"(tell me a|do you know a|share a|give me a) (fun|interesting|cool) (fact|facts)",
     ["Octopuses have three hearts, nine brains, and blue blood!",
      "A group of flamingos is called a 'flamboyance'.",
      "The shortest war in history was between Britain and Zanzibar on August 27, 1896. Zanzibar surrendered after 38 minutes.",
      "Bananas are berries, but strawberries aren't!"
     ]],
     
    # Mathematics
    [r"(calculate|what is|what's) (\d+)\s*([+\-*/])\s*(\d+)",
     [lambda match: calculate_expression(match), "I can help with calculations! Try asking something like 'What is 5 + 3?' or 'Calculate 10 * 7'."],
     ],
     
    [r"(square|square root|cube|cube root|factorial) of (\d+)",
     [lambda match: calculate_math_operation(match), "I can help with various math operations! Try asking 'What is the square of 5?' or 'Calculate factorial of 6'."],
     ],
     
    [r"(prime|is \d+ a prime number|check if \d+ is prime)",
     [lambda match: check_prime(match), "I can check if numbers are prime! Try asking 'Is 17 a prime number?' or 'Check if 21 is prime'."],
     ],
     
    # General Knowledge
    [r"(who is|tell me about) (the president|prime minister) of (india|usa|uk|australia|canada|japan|china|russia|germany|france)",
     [lambda match: get_leader_info(match), "I can tell you about world leaders! Try asking 'Who is the president of the USA?' or 'Tell me about the Prime Minister of India'."],
     ],
     
    [r"(capital of|what is the capital of) (\w+)",
     [lambda match: get_capital(match), "I know world capitals! Try asking 'What is the capital of France?' or 'Capital of Japan'."],
     ],
     
    [r"(population of|what is the population of) (\w+)",
     [lambda match: get_population(match), "I have population data! Try asking 'What is the population of China?' or 'Population of Brazil'."],
     ],
     
    # History
    [r"(tell me about|who was) (mahatma gandhi|albert einstein|isaac newton|marie curie|leonardo da vinci|nikola tesla)",
     [lambda match: get_historical_figure(match), "I know about historical figures! Try asking 'Tell me about Albert Einstein' or 'Who was Mahatma Gandhi?'"],
     ],
     
    [r"(when was|in what year) (world war i|world war ii|the american revolution|the french revolution|the industrial revolution)",
     [lambda match: get_historical_event(match), "I know about historical events! Try asking 'When was World War II?' or 'In what year was the Industrial Revolution?'"],
     ],
     
    # Science and Technology
    [r"(tell me a|do you know a|share a|give me a) (science|scientific) (fact|facts)",
     [lambda match: get_science_fact(match), "I can share interesting science facts! Try asking 'Tell me a science fact' or 'Give me a scientific fact'."],
     ],
     
    [r"(tell me a|do you know a|share a|give me a) (tech|technology|computer) (fact|facts)",
     [lambda match: get_tech_fact(match), "I know lots of tech facts! Try asking 'Tell me a tech fact' or 'Share a computer fact'."],
     ],
     
    [r"(tell me a|do you know a|share a|give me a) (space|astronomy|cosmic) (fact|facts)",
     [lambda match: get_space_fact(match), "I love space facts! Try asking 'Tell me a space fact' or 'Share an astronomy fact'."],
     ],
     
    # Current Affairs (as of knowledge cutoff in 2023)
    [r"(what'?s new|latest news|current affairs|recent developments)",
     ["As of my last update in 2023, here are some notable developments:\n- AI and machine learning continue to advance rapidly\n- Renewable energy adoption is increasing globally\n- Space exploration is seeing renewed interest with missions to the Moon and Mars\n- Climate change remains a critical global challenge\n\nFor the most current news, I recommend checking a reliable news source.",
      "I can share that as of 2023, there's significant progress in AI, renewable energy, and space exploration. However, for the very latest updates, I'd recommend checking a news website or app."
     ]],
     
    # Default response
    [r"(.*)",
     ["I'm not sure I understand. Could you rephrase that?",
      "That's an interesting point! Could you tell me more about what you're interested in?",
      "I'm still learning. Could you try asking something else? Maybe about science, technology, or current affairs?",
      "I don't have an answer for that, but I'm here to chat! What else would you like to know?"]],
    
    # Default response
    [r"(.*)",
     ["I'm not sure I understand. Could you rephrase that?",
      "Interesting! Tell me more.",
      "I'm still learning. Could you try asking something else?",
      "I don't have an answer for that, but I'm here to chat!",
      "Let me think about that... How about we talk about something else?"]]
]

# Initialize chatbot
chatbot = Chat(chat_pairs, reflections)

# Route to serve generated images
@app.route('/generated_images/<filename>')
def serve_generated_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Default route
@app.route('/')
def home():
    return "AI Showcase API is running!"

def generate_user_id():
    """Generate a unique user ID for session tracking."""
    return str(random.getrandbits(64)) + str(int(datetime.now().timestamp()))

def analyze_sentiment(text):
    """Analyze the sentiment of the input text."""
    if not text or not isinstance(text, str):
        return 'neutral', {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
    try:
        scores = sia.polarity_scores(text)
        sentiment = 'neutral'
        if scores['compound'] > 0.05:
            sentiment = 'positive'
        elif scores['compound'] < -0.05:
            sentiment = 'negative'
        return sentiment, scores
    except Exception as e:
        print(f"Error in analyze_sentiment: {str(e)}")
        return 'neutral', {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

def extract_entities(text):
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def detect_intent(text):
    """Detect the intent of the user's message."""
    text = text.lower()
    intents = {
        'greeting': any(word in text for word in ['hi', 'hello', 'hey', 'greetings']),
        'goodbye': any(word in text for word in ['bye', 'goodbye', 'see you', 'farewell']),
        'thanks': any(word in text for word in ['thank', 'thanks', 'appreciate']),
        'help': any(word in text for word in ['help', 'support', 'assist']),
        'joke': any(word in text for word in ['joke', 'funny', 'laugh']),
        'weather': any(word in text for word in ['weather', 'temperature', 'forecast']),
        'search': any(word in text for word in ['search', 'find', 'look up', 'what is']),
        'remember': 'remember that' in text,
        'repeat': any(word in text for word in ['repeat', 'say again', 'what did you say'])
    }
    return next((intent for intent, is_present in intents.items() if is_present), 'unknown')

def get_wikipedia_summary(query, sentences=2):
    """Get a summary from Wikipedia."""
    try:
        page = wiki_wiki.page(query)
        if page.exists():
            return page.summary[:500] + '...' if len(page.summary) > 500 else page.summary
    except:
        pass
    return None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'response': 'Please provide a message'}), 400
            
        message = str(data.get('message', '')).strip()
        if not message:
            return jsonify({'response': 'Message cannot be empty'}), 400
            
        print(f"\n=== New Request ===")
        print(f"Received message: {message}")
        
        # Enhanced response patterns
        message_lower = message.lower().strip()
        
        # Greetings
        if any(greeting in message_lower for greeting in ['hi', 'hello', 'hey', 'greetings', 'hi there']):
            greetings = [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Greetings! How may I be of service?",
                "Hello! What would you like to know?"
            ]
            response = random.choice(greetings)
            
        # Self-identification
        elif any(phrase in message_lower for phrase in ['who are you', 'what are you', 'your name']):
            response = "I'm your AI assistant, here to help answer your questions and assist with various tasks. You can ask me about general knowledge, calculations, or just chat!"
        
        # How are you
        elif 'how are you' in message_lower:
            responses = [
                "I'm just a computer program, but I'm functioning perfectly! How can I assist you today?",
                "I don't have feelings, but I'm here and ready to help! What can I do for you?",
                "I'm running smoothly! How can I be of service?"
            ]
            response = random.choice(responses)
        
        # Gratitude
        elif any(word in message_lower for word in ['thank', 'thanks', 'appreciate']):
            responses = [
                "You're welcome! Is there anything else I can help you with?",
                "My pleasure! Feel free to ask if you need anything else.",
                "You're welcome! Don't hesitate to ask if you have more questions."
            ]
            response = random.choice(responses)
        
        # Goodbye
        elif any(word in message_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            responses = [
                "Goodbye! Have a wonderful day!",
                "Farewell! Come back if you have more questions!",
                "Goodbye! It was nice chatting with you!"
            ]
            response = random.choice(responses)
        
        # Math operations
        elif any(op in message_lower for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'multiplied by', 'divided by', '=']):
            try:
                # Helper function to extract numbers from text
                def extract_numbers(text):
                    import re
                    # Find all numbers in the text, including decimals and negative numbers
                    numbers = re.findall(r'-?\d+\.?\d*', text)
                    # Convert to float and then to int if it's a whole number
                    return [int(float(num)) if float(num).is_integer() else float(num) for num in numbers]
                
                # Extract all numbers from the message
                numbers = extract_numbers(message)
                
                # Handle word-based math first
                if 'plus' in message_lower or 'add' in message_lower:
                    if len(numbers) >= 2:
                        response = f"{numbers[0]} plus {numbers[1]} equals {numbers[0] + numbers[1]}."
                    elif len(numbers) == 1:
                        response = f"I need another number to add to {numbers[0]}. For example: 'What is {numbers[0]} plus 3?'"
                    else:
                        response = "I need two numbers to add. For example: 'What is 5 plus 3?'"
                
                elif 'minus' in message_lower or 'subtract' in message_lower:
                    if len(numbers) >= 2:
                        response = f"{numbers[0]} minus {numbers[1]} equals {numbers[0] - numbers[1]}."
                    elif len(numbers) == 1:
                        response = f"I need another number to subtract from {numbers[0]}. For example: 'What is {numbers[0]} minus 2?'"
                    else:
                        response = "I need two numbers to subtract. For example: 'What is 10 minus 4?'"
                
                elif 'times' in message_lower or 'multiplied by' in message_lower or 'multiply' in message_lower:
                    if len(numbers) >= 2:
                        response = f"{numbers[0]} times {numbers[1]} equals {numbers[0] * numbers[1]}."
                    elif len(numbers) == 1:
                        response = f"I need another number to multiply by {numbers[0]}. For example: 'What is {numbers[0]} times 5?'"
                    else:
                        response = "I need two numbers to multiply. For example: 'What is 6 times 7?'"
                
                elif 'divided by' in message_lower or 'divide' in message_lower:
                    if len(numbers) >= 2:
                        if numbers[1] != 0:
                            result = numbers[0] / numbers[1]
                            if isinstance(result, float) and result.is_integer():
                                result = int(result)
                            response = f"{numbers[0]} divided by {numbers[1]} equals {result}."
                        else:
                            response = "I can't divide by zero!"
                    elif len(numbers) == 1:
                        response = f"I need another number to divide {numbers[0]} by. For example: 'What is {numbers[0]} divided by 2?'"
                    else:
                        response = "I need two numbers to divide. For example: 'What is 10 divided by 2?'"
                
                # Handle symbolic math operations (e.g., 2+2, 5*5)
                elif any(op in message for op in ['+', '-', '*', '/']):
                    # Extract the mathematical expression
                    expr = ''.join(c for c in message if c in '0123456789+-*/.() ')
                    expr = expr.strip()
                    if expr:  # Make sure we have something to evaluate
                        try:
                            # Use a safer evaluation method
                            result = eval(expr, {"__builtins__": None}, {})
                            response = f"The result of {expr} is {result}."
                        except:
                            response = "I couldn't process that calculation. Could you rephrase it?"
                    else:
                        response = "I need a valid mathematical expression. For example: 'What is 2+2?'"
                
                else:
                    response = "I'm not sure about that calculation. Could you rephrase it?"
                    
            except Exception as e:
                print(f"Math error: {str(e)}")
                response = "I had trouble with that calculation. Could you try rephrasing it?"
        
        # General knowledge
        elif 'states in india' in message_lower or 'how many states in india' in message_lower:
            response = "India has 28 states and 8 Union Territories as of my last update in 2023."
            
        elif 'capital of india' in message_lower:
            response = "The capital of India is New Delhi."
            
        elif 'population of india' in message_lower:
            response = "As of 2023, India's population is approximately 1.4 billion people, making it the most populous country in the world."
        
        # Current date and time
        elif any(phrase in message_lower for phrase in ['what time is it', 'current time', 'what is the time']):
            from datetime import datetime
            current_time = datetime.now().strftime("%I:%M %p")
            response = f"The current time is {current_time}."
            
        elif any(phrase in message_lower for phrase in ['what day is it', 'what is today', 'current date']):
            from datetime import datetime
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            response = f"Today is {current_date}."
        
        # Check for image generation request
        elif any(phrase in message_lower for phrase in ['generate image', 'create image', 'make a picture', 'draw a picture', 'generate photo']):
            # Extract the prompt from the message
            prompt = message
            for phrase in ['generate image of', 'create image of', 'make a picture of', 'draw a picture of', 'generate photo of', 'generate an image of', 'create an image of']:
                if phrase in message_lower:
                    prompt = message[message_lower.find(phrase) + len(phrase):].strip()
                    break
            
            if not prompt or prompt == message:  # If we couldn't extract a specific prompt
                return jsonify({
                    'type': 'text',
                    'content': "I'd be happy to generate an image for you! Could you please describe what you'd like me to create? For example: 'Generate an image of a sunset over mountains'"
                })
            else:
                try:
                    # Generate image using DALL-E
                    image_response = generate_image(prompt)
                    return jsonify(image_response)
                except Exception as e:
                    print(f"Error generating image: {str(e)}")
                    return jsonify({
                        'type': 'text',
                        'content': "I encountered an error while generating the image. Please try again later."
                    })
        
        # Default response for unknown queries
        else:
            responses = [
                f"I'm not sure how to respond to '{message}'. Could you try asking something else?",
                f"That's an interesting question! I'm still learning, but you asked: '{message}'",
                f"I'm not certain about that. Could you rephrase your question about '{message}'?",
                f"I'm still learning about these things. Could you tell me more about what you mean by '{message}'?"
            ]
            response = random.choice(responses)
        
        print(f"Sending response: {response}")
        return jsonify({
            'type': 'text',
            'content': response
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Print full traceback for debugging
        return jsonify({'response': 'Sorry, there was an error processing your request. Please try again.'}), 500

@app.route('/api/recommend', methods=['GET'])
def recommend():
    movies = ["The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction", "Forrest Gump"]
    recommended = random.sample(movies, min(3, len(movies)))
    return jsonify({'recommendations': recommended})

def generate_image(prompt, width=512, height=512, num_inference_steps=30, guidance_scale=7.5):
    """
    Generate an image using Stable Diffusion
    
    Args:
        prompt (str): Text description of the desired image
        width (int): Width of the generated image (default: 512)
        height (int): Height of the generated image (default: 512)
        num_inference_steps (int): Number of denoising steps (default: 30)
        guidance_scale (float): Guidance scale (default: 7.5)
        
    Returns:
        dict: Dictionary containing image URL and prompt or error message
    """
    if pipe is None:
        return {
            "type": "text",
            "content": "Image generation model is not available. Please check the server logs for errors."
        }
    
    try:
        print(f"Generating image with prompt: {prompt}")
        
        # Generate the image
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        
        # Generate a unique filename
        filename = f"{str(uuid.uuid4())}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the image
        image.save(filepath)
        
        # Return the image URL
        image_url = f"/generated_images/{filename}"
        return {
            "type": "image",
            "url": image_url,
            "prompt": prompt
        }
        
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        return {
            "type": "text",
            "content": f"I encountered an error while generating the image: {str(e)}"
        }

if __name__ == '__main__':
    print("\nStarting AI Showcase server on port 5001...")
    print("Access it at: http://localhost:5001")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=True, port=5001)
