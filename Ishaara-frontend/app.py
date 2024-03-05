from langdetect import detect
from mtranslate import translate
import requests
import speech_recognition as sr
from googletrans import Translator

from preprocess import preprocess
from flask import Flask,request,render_template,send_from_directory,jsonify

app =Flask(__name__,static_folder='static', static_url_path='')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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

def translate_to_english(text, target_language, current_language):
    try:
        translated_text = translate(text, target_language, current_language)
        return translated_text
    except Exception as e:
        return str(e)

@app.route('/',methods=['GET'])
def text():
	# clear_all();
	return render_template('encode-text.html')

@app.route('/audio/',methods=['GET'])
def audio():
    # clear_all();
    return render_template('encode-audio.html')

# serve sigml files for animation
@app.route('/static/<path:path>')
def serve_signfiles(path):
	print("here");
	return send_from_directory('static',path)

@app.route('/translate', methods=['POST'])
def trans():
	data = request.json  # Get data from the request
    # Extract variables from the data
	text = data['text']
	curr_lang=detect(text)
	if curr_lang!='en':
		text = translate_to_english(text, 'en', curr_lang)
		preprocess_text=preprocess(text)
	else:
		preprocess_text = text
    # Prepare the response
	response_data = {
        'status': 'success',
        'preprocess_text': preprocess_text,
    }
	print(response_data)
    
	return jsonify(response_data)

@app.route('/transcript', methods=['POST'])
def transcript():
    if request.method=='POST':
        lang=request.form.get('lang')
        print(lang)
        if lang!='None' or lang!=None:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something")
                audio = r.listen(source)

            try:
                text = r.recognize_google(audio, language='en-IN')
                print("You said:", text)

                translated_text = translate_to_english(text, 'en', lang)
                print("Translated text (English):", translated_text)
                preprocess_text=preprocess(translated_text)
                return render_template('encode-audio.html', text=text, lang=lang, ptext=preprocess_text)
            except:
                text=""
                return render_template('encode-audio.html')


if __name__=="__main__":
	app.run(debug=True)