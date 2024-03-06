from flask import Flask, render_template, request, redirect, url_for, flash
from langdetect import detect
from mtranslate import translate
import requests
import speech_recognition as sr
from googletrans import Translator

from preprocess import preprocess, remove_punctuation
from dataset_generator.extract_features import process_video
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

@app.route('/encode/text/', methods=['GET', 'POST'])
def encoder():
    if request.method=='POST':
        text=request.form.get('text')
        text=remove_punctuation(text)
        curr_lang=detect(text)
        if curr_lang!='en':
            text = translate_to_english(text, 'en', curr_lang)
        preprocess_text=preprocess(text)
        return render_template('encode-text.html', text=text, lang=curr_lang, ptext=preprocess_text)
    return render_template('encode-text.html')

@app.route('/encode/voice/', methods=['GET', 'POST'])
def voice():
    if request.method=='POST':
        lang=request.form.get('lang')
        print(lang)
        if lang!='None' or lang!=None:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something")
                flash('Recognizing speech')
                audio = r.listen(source)

            try:
                text = r.recognize_google(audio, language=languages[lang])
                print("You said:", text)

                if lang!='English':
                    translated_text = translate_to_english(text, 'en', lang)
                    print("Translated text (English):", translated_text)
                preprocess_text=preprocess(text)
                return render_template('encode-text.html', text=text, lang=lang, ptext=preprocess_text)
            except:
                text=""
                return render_template('encode-voice.html')
        
    return render_template('encode-voice.html')

@app.route('/encode/file/', methods=['GET', 'POST'])
def file_input():
    if request.method=='POST':
        if 'fileInput' not in request.files:
            return 'No file part'

        file = request.files['fileInput']
        if file:
            # file.save('/path/to/save/' + file.filename)
            print('File uploaded successfully')
    return render_template('encode-file.html')

# @app.route('/decode/')
# def video_input():
    # final_list=process_video(0)
    # print(len(final_list))
    # print(final_list)
    
      
    
if __name__=='__main__':
    app.run(debug=True)