from labelers.gpt_labeler import GPTLabeler
from utils.std import *
from utils.config import *
import requests
import datetime

data = json.load(open('test_data/reid_raw.json'))

texts = []
images = []

for i in data:
    print(i['id'])
    texts += i['captions']
    images.append('test_data/' + i['file_path'])

labeler = GPTLabeler()

ans = [[labeler.label(i, j) for j in texts] for i in images]

for i in ans:
    print(i)