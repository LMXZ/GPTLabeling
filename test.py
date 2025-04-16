from utils.std import *
from utils.config import *
import requests
import datetime
from labelers.ds_labeler import DeepSeekLabeler
from labelers.gpt_labeler import GPTLabeler

labeler = DeepSeekLabeler()

data = json.load(open('test_data/reid_raw.json'))[:8]

images = []
texts = []

for i in data:
    images += ['test_data/' + i['file_path']]
    texts += i['captions']

labeler = GPTLabeler()

ans = [labeler.label(i, texts) for i in images]

for i in ans:
    print(i)

