import os
import json

words=[]
f=open('words.txt', 'w')
for folder in os.listdir('dataset_generator/words/'):
    words.append(folder)
    f.write(folder+'\n')

label_encoding={label:num for num,label in enumerate(words)}

with open('data.json', 'w') as f:
    json.dump(label_encoding, f)