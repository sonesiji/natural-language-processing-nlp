import streamlit as st
import requests
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Initialize SpaCy model
nlp = spacy.load("en_core_web_sm")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Define a function to fetch a news article based on a headline
def fetch_news_article(api_key, headline):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': headline,
        'apiKey': api_key,
        'pageSize': 1
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        
        if data.get('status') == 'ok' and data['articles']:
            article = data['articles'][0]
            title = article['title']
            content = article['content']
            return title, content
        else:
            st.error("No articles found or API request failed.")
            return None, None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None, None

# Define functions to extract named entities
def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_entities_nltk(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    chunks = ne_chunk(pos_tags)
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity = ' '.join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))
    return entities

def compare_entities(spacy_ents, nltk_ents):
    spacy_set = set(spacy_ents)
    nltk_set = set(nltk_ents)
    
    common_ents = spacy_set & nltk_set
    spacy_unique = spacy_set - nltk_set
    nltk_unique = nltk_set - spacy_set
    
    return common_ents, spacy_unique, nltk_unique

# Streamlit app
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .main:hover {
        transform: scale(1.02);
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #343a40;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #495057;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .content {
        font-size: 18px;
        padding: 15px;
        color: #212529;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .entities {
        font-size: 18px;
        padding: 15px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 10px 0;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stTextInput > div > div > input {
        text-align: center;
        background-color: #e9ecef;
        border: none;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        outline: none;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">Checking Each Named Entity Recognition Comparison</div>', unsafe_allow_html=True)

# Center the search bar
st.markdown('<div class="center">', unsafe_allow_html=True)

# Sidebar for user input
headline = st.text_input("Enter News You Want to Know about", "")

st.markdown('</div>', unsafe_allow_html=True)

# API key (set your own API key here)
api_key = 'e83e0f9cdb4c4864b1dae64dc627b83e'

if headline:
    st.write(f"<div class='subheader'>Analysing news for headline:</div>", unsafe_allow_html=True)
    
    title, content = fetch_news_article(api_key, headline)
    
    if content:
        st.markdown(f'<div class="subheader">Title:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="subheader">Content:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="content">{content}</div>', unsafe_allow_html=True)

        # Extract entities
        spacy_entities = extract_entities_spacy(content)
        nltk_entities = extract_entities_nltk(content)

        st.markdown(f'<div class="subheader">SpaCy Entities</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entities">{spacy_entities}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="subheader">NLTK Entities</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entities">{nltk_entities}</div>', unsafe_allow_html=True)

        # Compare entities
        common_ents, spacy_unique, nltk_unique = compare_entities(spacy_entities, nltk_entities)

        st.markdown(f'<div class="subheader">Comparison Results</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entities">**Common Entities:** {common_ents}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entities">**SpaCy Unique Entities:** {spacy_unique}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entities">**NLTK Unique Entities:** {nltk_unique}</div>', unsafe_allow_html=True)
    else:
        st.write("No content found for the given headline.")
