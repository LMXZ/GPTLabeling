from labelers.gpt_labeler import GPTLabeler
from utils.std import *
from utils.config import *
import requests
import datetime

data = json.load(open('test_data/reid_raw.json'))[:8]

texts = []
images = []

for i in data:
    print(i['id'])
    texts += i['captions']
    images.append('test_data/' + i['file_path'])

labeler = GPTLabeler()

ans = [labeler.label(i, texts) for i in images]

for i in ans:
    print(i)