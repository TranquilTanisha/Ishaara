import requests

api_key = 'your_ginger_api_key'
text = "This are example sentence with bad grammar"
url = f"https://services.gingersoftware.com/GrammarApi/grammar?text={text}&apiKey={api_key}&lang=US"

response = requests.get(url)
result = response.json()

corrected_text = text
for suggestion in reversed(result['Corrections']):
    start = suggestion['From']
    end = suggestion['To']
    replacement = suggestion['Suggestions'][0]['Text']
    corrected_text = corrected_text[:start] + replacement + corrected_text[end:]

print("Original Sentence: ", text)
print("Corrected Sentence: ", corrected_text)
