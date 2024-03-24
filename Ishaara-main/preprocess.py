import spacy
# python -m spacy download en_core_web_sm
from string import punctuation

nlp=spacy.load('en_core_web_sm')
stopwords=nlp.Defaults.stop_words

def preprocess(text):
    text = text.lower()
    doc=nlp(text)
    tokens=[token for token in doc]
    for token in tokens:
        if token in stopwords or str(token) in punctuation:
            tokens.remove(token)
    tokens = [token.lemma_ for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
