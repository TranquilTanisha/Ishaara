from flask import Flask,request,render_template,send_from_directory,jsonify, redirect

from langdetect import detect
from mtranslate import translate
import requests
import speech_recognition as sr

from capture_video_LSTM import capture_video
from preprocess import preprocess

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

def check_internet_connection():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

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

@app.route('/',methods=['GET'])
def text():
	# clear_all();
	return render_template('encode-text.html')

@app.route('/audio/',methods=['GET'])
def audio():
    # clear_all();
    return render_template('encode-audio.html')

@app.route('/video/',methods=['GET'])
def video():
    return render_template('decode-video.html')

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

@app.route('/transcript', methods=['GET', 'POST'])
def transcript():
    print(request.method)
    if request.method=='POST':
        lang=request.form.get('lang')
        print(lang)
        if lang!='None' or lang!=None:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Say something")
                audio = r.listen(source)

            try:
                text = r.recognize_google(audio, language=languages[lang])
                print("You said:", text)

                if lang!='English':
                    text = translate_to_english(text, 'en', lang)
                    print("Translated text (English):", text)
                preprocess_text=preprocess(text)
                return render_template('encode-audio.html', text=text, lang=lang, ptext=preprocess_text)
            except:
                text="Did not detect anything"
                return redirect('/audio/')
        else:
            return redirect('/audio/')

@app.route('/recordvid/', methods=['GET', 'POST'])
def recordvid():
    if request.method=='POST':
        lang=request.form.get('lang')
        print(lang)
        if lang!='None' or lang!=None:
            res=capture_video(lang)
            print(res)
            if check_internet_connection():
                res=correct_grammar_with_languagetool(res)
            tr=''
            for r in res:
                if lang!='English':
                    r = translate_to_english(r, languages[lang].split('-')[0], 'en')
                tr+=r+' '
            print(tr)
            return render_template('decode-video.html', res=tr, lang=lang)
        else:
            return redirect('/video/')
        
    return redirect('/video/')

if __name__=="__main__":
	app.run(debug=False)