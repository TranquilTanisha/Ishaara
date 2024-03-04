import spacy
nlp=spacy.load('en_core_web_sm')
stopwords=nlp.Defaults.stop_words
from string import punctuation

def remove_punctuation(text):
    return ''.join([char for char in text if char not in punctuation])

def preprocess(text):
    text = text.lower()
    doc=nlp(text)
    tokens=[token for token in doc]
    for token in tokens:
        if token in stopwords:
            tokens.remove(token)
    tokens = [token.lemma_ for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
