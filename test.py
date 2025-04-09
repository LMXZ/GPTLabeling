from labelers.gpt_labeler import GPTLabeler, GPTCoTLabeler
from utils.std import *
from utils.config import *
import requests
import datetime

data = json.load(open('dataset/CUHK-PEDES/reid_raw.json'))

dd = DefaultDict(lambda : 0)
dd2 = DefaultDict(lambda : 0)

for i in data:
    dd[i['id']] += 1

for k, v in dd.items():
    dd2[v]+=1

print(dd2)
exit()

data = json.load(open('test_data/reid_raw.json'))[-6:]

texts = []
images = []

for i in data:
    texts += i['captions']
    images.append('test_data/' + i['file_path'])

for i in texts:
    print(i)

for i in images:
    print(i)

labeler = GPTCoTLabeler()

print(labeler.label(images, texts))
