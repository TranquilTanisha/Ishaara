import pyttsx3

speaker=pyttsx3.init()
voices=speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 120)

speaker.say('Hello')
speaker.runAndWait()