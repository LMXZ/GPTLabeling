from utils.std import *
from utils.config import *
import requests
import datetime
from labelers.ds_labeler import DeepSeekLabeler

labeler = DeepSeekLabeler()

data = json.load(open('test_data/reid_raw.json'))

images = []
texts = []

for i in data:
    images += ['test_data/' + i['file_path']]
    texts += i['captions']

ans = []
for ii, i in enumerate(images):
    ans += [[]]
    for jj, j in enumerate(texts):
        s = labeler.label(i, j)
        ans[-1] += [s]
        print('-' * 10, ii, ' ', jj, s)

for i in ans:
    print(i)

