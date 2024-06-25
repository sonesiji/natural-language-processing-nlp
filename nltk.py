import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Input text from user
text = "NLTK is a leading platform for building Python programs to work with human language data. It's very useful for text processing"

# Tokenize the text
words = word_tokenize(text)
print("\nOriginal Words:", words)

# Initialize the Porter Stemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("\nStemmed Words:", stemmed_words)


# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("\nLemmatized Words:", lemmatized_words)

# Remove Stop Words
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("\nFiltered Words:", filtered_words)