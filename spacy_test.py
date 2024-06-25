import spacy

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Hello there! How are you doing today? This is a sample text for tokenization, stemming, lemmatization, and stopword removal."

# Process the text with SpaCy
doc = nlp(text)

# Sentence tokenization
print("Sentences:")
for sent in doc.sents:
    print(sent.text)

# Word tokenization and stopword removal
filtered_words = [token for token in doc if not token.is_stop and not token.is_punct]
print("\nFiltered Words (Stopword Removal):")
print([token.text for token in filtered_words])

# Stemming (using lemmatizer as a proxy)
print("\nStemmed Words (using Lemmatizer as proxy):")
print([token.lemma_ for token in filtered_words])

# Lemmatization
print("\nLemmatized Words:")
print([token.lemma_ for token in filtered_words])