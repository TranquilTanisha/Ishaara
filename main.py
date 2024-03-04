from flask import Flask, render_template, request, redirect, url_for
from langdetect import detect
from mtranslate import translate
import requests
import speech_recognition as sr
from googletrans import Translator

from preprocess import preprocess
# from encoder import encoder

app=Flask(__name__)
app.secret_key='secret'

languages={'Hindi': 'hi-IN',
'Bengali': 'bn-IN',
'Telugu': 'te-IN',
'Marathi': 'mr-IN',
'Tamil': 'ta-IN',
'Urdu': 'ur-IN',
'Gujarati': 'gu-IN',
'Malayalam': 'ml-IN',
'Kannada': 'kn-IN',
'Odia': 'or-IN',
'Punjabi': 'pa-IN',
'Assamese': 'as-IN'}

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
        translated_text = translate(text, target_language, current_language)
        return translated_text
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/encode/', methods=['GET', 'POST'])
def encoder():
    if request.method=='POST':
        text=request.form.get('text')
        curr_lang=detect(text)
        if curr_lang!='en':
            text = translate_to_english(text, 'en', curr_lang)
        preprocess_text=preprocess(text)
        return render_template('encode.html', text=text, lang=curr_lang, ptext=preprocess_text)
    return render_template('encode.html')

@app.route('/encode/', methods=['GET', 'POST'])
def voice():
    if request.method=='POST':
        if request.form.get('type')=='voice':
            lang=request.form.get('lang')
            print(lang)
            if lang!='None' or lang!=None:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Say something")
                    audio = r.listen(source)

                try:
                    text = r.recognize_google(audio, language='hi-IN')
                    print("You said:", text)

                    translated_text = translate_to_english(text, 'en', lang)
                    print("Translated text (English):", translated_text)
                    preprocess_text=preprocess(text)
                    return render_template('encode.html', text=text, lang=lang, ptext=preprocess_text)
                except:
                    text=""
                    return render_template('encode.html')
        elif request.form.get('type')=='text':
            text=request.form.get('text')
            curr_lang=detect(text)
            if curr_lang!='en':
                text = translate_to_english(text, 'en', curr_lang)
            preprocess_text=preprocess(text)
            return render_template('encode.html', text=text, lang=curr_lang, ptext=preprocess_text)

    return render_template('encode.html')
    
if __name__=='__main__':
    app.run(debug=True)
