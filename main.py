from flask import Flask, render_template, request, redirect, url_for
from langdetect import detect
from mtranslate import translate
import requests

from preprocess import preprocess
# from encoder import encoder

app=Flask(__name__)
app.secret_key='secret'

#for english language only
def correct_grammar_with_languagetool(text):
    # global translated_text
    api_url = "https://api.languagetool.org/v2/check"
    payload = {"text": text, "language": "en-US"}

    response = requests.post(api_url, data=payload)
    grammar_errors = response.json()["matches"]

    corrected_text = text
    for error in grammar_errors:
        corrected_text = corrected_text[:error['offset']] + error['replacements'][0]['value'] + corrected_text[error['offset'] + error['length']:]

    return corrected_text

def translate_to_english(text, target_language, current_language):
    try:
        # translated_text = translate(text, target_language, current_language)
        # text=correct_grammar_with_languagetool(text)
        translated_text = translate(text, target_language, current_language)
        return translated_text
    except Exception as e:
        return str(e)

@app.route('/encoding', methods=['GET', 'POST'])
def encoder():
    if request.method=='POST':
        text=request.form.get('text')
        curr_lang=detect(text)
        if curr_lang!='en':
            text = translate_to_english(text, 'en', curr_lang)
        preprocess_text=preprocess(text)
        return render_template('encoder.html', text=text, lang=curr_lang, ptext=preprocess_text)
    return render_template('encoder.html')

if __name__=='__main__':
    app.run(debug=True)
