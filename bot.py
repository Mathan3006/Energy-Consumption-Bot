
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import random
import numpy as np
import re

# Initialize Flask app
app = Flask(__name__)

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Initialize conversation context
conversation_context = {}

# Preprocessing functions
def preprocess_text(text):
    """ Preprocess the input text: tokenize, remove stop words, and lemmatize """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stop words and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# Sentiment Analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# Named Entity Recognition (NER) using spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Context Management
def update_context(user_id, response):
    """ Update the context with the user's latest message and bot response """
    if user_id not in conversation_context:
        conversation_context[user_id] = {'messages': []}
    
    conversation_context[user_id]['messages'].append(response)

def get_context(user_id):
    """ Retrieve the context for a particular user """
    if user_id in conversation_context:
        return conversation_context[user_id]['messages']
    return []

# More advanced dynamic response generation based on context, sentiment, and entities
def generate_response(user_message, user_id):
    """ Generate a dynamic response based on user message and context """
    
    # Step 1: Preprocess the user message
    tokens = preprocess_text(user_message)
    
    # Step 2: Perform Sentiment Analysis
    sentiment = analyze_sentiment(user_message)
    
    # Step 3: Extract Named Entities (if any)
    entities = extract_entities(user_message)

    # Step 4: Retrieve context from the conversation (previous messages)
    context = get_context(user_id)
    
    # Step 5: Generate response
    if sentiment == 'positive':
        response = "It sounds like you're in a good mood! How can I help?"
    elif sentiment == 'negative':
        response = "I'm sorry to hear that. How can I assist you?"
    else:
        response = "Got it. How can I help you today?"

    if entities:
        response += f" I noticed you mentioned "
        response += ", ".join([f"{ent[0]} ({ent[1]})" for ent in entities])
        response += "."
    
    # Use context for more personalized responses
    if context:
        last_message = context[-1]
        response += f" By the way, last time you mentioned {last_message}."
    
    # Update the context for the user
    update_context(user_id, response)
    
    return response

# Handling ambiguous user inputs
def handle_ambiguity(user_message):
    """ Handle ambiguous user input by asking clarification questions """
    ambiguous_keywords = ['what', 'how', 'why']
    if any(keyword in user_message.lower() for keyword in ambiguous_keywords):
        return "Could you please clarify what you mean?"
    return ""

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Extract the user input from the POST request
        data = request.json
        user_message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default_user')  # Unique ID for each user to maintain context

        if not user_message:
            return jsonify({'reply': "Please enter a valid message.", 'sentiment': 'neutral', 'entities': []})

        # Handle ambiguity in the message
        ambiguity_response = handle_ambiguity(user_message)
        if ambiguity_response:
            return jsonify({'reply': ambiguity_response, 'sentiment': 'neutral', 'entities': []})

        # Generate a dynamic response based on the user message
        bot_response = generate_response(user_message, user_id)
        
        # Perform Sentiment Analysis and Entity Extraction
        sentiment = analyze_sentiment(user_message)
        entities = extract_entities(user_message)
        
        # Return the bot's response along with sentiment and entities
        return jsonify({
            'reply': bot_response,
            'sentiment': sentiment,
            'entities': entities
        })

    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'reply': "Something went wrong. Please try again later.", 'sentiment': 'neutral', 'entities': []}), 500

if __name__ == '__main__':
    app.run(debug=True)
 