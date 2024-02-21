import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

import spacy
nlp=spacy.load('en_core_web_sm')
stopwords=nlp.Defaults.stop_words
from string import punctuation

def preprocess(text):
    text = text.lower()
    doc=nlp(text)
    tokens=[token.text for token in doc]
    print(tokens)
    # tokens = word_tokenize(text)
    # tokens = [token for token in tokens if token not in nltk.punkt]
   # stop_words = set(stopwords.words('english'))
    #tokens = [token for token in tokens if token not in stop_words]
    for token in tokens:
        if token in stopwords or token in punctuation:
            tokens.remove(token)
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
