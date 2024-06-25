import nltk
nltk.download('punkt', quiet=True)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

text = input("Enter the text: ")
print([PorterStemmer().stem(token) for token in word_tokenize(text)])
