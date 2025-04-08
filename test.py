from labelers.gpt_labeler import GPTLabeler, GPTCoTLabeler
from utils.std import *
from utils.config import *
import requests
import datetime

data = json.load(open('test_data/reid_raw.json'))[-4:]

texts = []
images = []

for i in data:
    texts += i['captions']
    images.append('test_data/' + i['file_path'])

# for i in texts:
#     print(i)

labeler = GPTCoTLabeler()

print(labeler.label(images[0], texts[0]))
print(labeler.label(images[0], texts[-1]))

print(labeler.label(images[-1], texts[0]))
print(labeler.label(images[-1], texts[-1]))

