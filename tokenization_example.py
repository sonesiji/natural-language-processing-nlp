# tokenization_example.py

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = input("enter the text")
tokens = word_tokenize(text)
print(tokens)
