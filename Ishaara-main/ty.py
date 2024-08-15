from gtts import gTTS
from playsound import playsound
import os

# Convert text to speech
text = "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"
language = "en"
speech = gTTS(text=text, lang=language, slow=False)

# Save and play the speech
speech.save("output.mp3")
playsound("output.mp3")
os.remove("output.mp3")
