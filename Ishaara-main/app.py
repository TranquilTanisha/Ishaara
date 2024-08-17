from flask import Flask,request,render_template,send_from_directory, redirect, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from langdetect import detect
import requests
import speech_recognition as sr
from capture_video_LSTM import capture_video
from preprocess import preprocess, return_languages, translate_to_english

app =Flask(__name__,static_folder='static', static_url_path='')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

languages=return_languages() # 108 languages

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Load the saved model and tokenizer
# save_directory = "./saved_grammar_model"
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# translation = AutoModelForSeq2SeqLM.from_pretrained(save_directory)

# def check_grammar(text):
#     input_ids = tokenizer(f"grammar: {text}", return_tensors="pt").input_ids
#     outputs = translation.generate(input_ids, max_new_tokens=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/',methods=['GET'])
def text():
    return render_template('encode-text.html')

@app.route('/audio/',methods=['GET'])
def audio():
    # clear_all();
    print(languages)
    return render_template('encode-audio.html', languages=languages)

@app.route('/video/',methods=['GET'])
def video():
    return render_template('decode-video.html')

# serve sigml files for animation
@app.route('/static/<path:path>')
def serve_signfiles(path):
	return send_from_directory('static',path)

@app.route('/translate', methods=['POST'])
def trans():
	data = request.json  # Get data from the request
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

                # if lang!='English':
                text = translate_to_english(text, 'en', lang)
                print("Translated text (English):", text)
                preprocess_text=preprocess(text)
                return render_template('encode-audio.html', text=text, lang=lang, ptext=preprocess_text, languages=languages)
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
            if len(res)==0: return render_template('decode-video.html', res='', lang=lang)
            if len(res)==0: return render_template('decode-video.html', res='', lang=lang)
            print(res)
            tr = ' '.join(word for word in res)
            # print(tr)
            # tr=check_grammar(tr)
            # print(res)
            # tr=translate_to_english(res, 'en', languages[lang].split('-')[0])
            # print(tr)
            return render_template('decode-video.html', res=tr, lang=lang)
        else:
            return redirect('/video/')
        
    return redirect('/video/')

if __name__=="__main__":
	app.run(debug=False)


