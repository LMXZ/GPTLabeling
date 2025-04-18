from utils.std import *
from utils.config import *
import requests
import datetime
from labelers.gpt_labeler import GPTLabeler
from utils.tasks import *
import threading
from threading import Thread
import lmdb
from utils.decos import NoValidAPIKey

thr_cnt = 10

# map_size定义最大储存容量，单位是kb，以下定义1TB容量 

# # 添加数据和键值 
# txn.put(key = '1', value = 'aaa') 
# txn.put(key = '2', value = 'bbb') 
# txn.put(key = '3', value = 'ccc') 
  
# # 通过键值删除数据 
# txn.delete(key = '1') 
  
# # 修改数据 
# txn.put(key = '3', value = 'ddd') 
  
# # 通过commit()函数提交更改 
# txn.commit() 
# env.close()

data = json.load(open(os.path.join(config['data_root'], 'reid_raw.json')))
match_info = open('match').read().split('\n')
match = defaultdict(lambda :-1)

id_cnt = len({i['id'] for i in data if i['split']=='train'})

for i in match_info:
    u, v = i.split(' ')
    u, v = int(u), int(v)
    match[u] = v
    match[v] = u

id_data: List[List] = []
for i in data:
    id = i['id'] - 1
    while len(id_data) - 1 < id:
        id_data.append([])
    id_data[id].append(i)

vis = set()
groups: List[TaskGroup] = []

def add(i):
    vis.add(i)
    for k in id_data[i]:
        groups[-1].images.append(os.path.join(config['data_root'], 'imgs',  k['file_path']))
        groups[-1].texts.extend(k['captions'])
        groups[-1].image_ids.append(k['id']-1)
        groups[-1].text_ids.extend([k['id']-1]*len(k['captions']))

for i in tqdm(range(id_cnt)):
    if i in vis:
        continue
    groups.append(TaskGroup([], [], [], []))
    j = match[i]
    add(i)
    if j != -1:
        add(j)

tasks = LabelingTask(groups, 'task_db')

js = []

def score(x: List[List[int]]):
    return [int((i[1] / (i[0] + i[1])) * 70) + int((min(8, i[1]) / 8) * 30) for i in x]

def work():
    labeler = GPTLabeler()
    while True:
        flag, task = tasks.get_task()
        if flag == -1:
            if task.image is not None:
                print('waiting...', task.image)
                time.sleep(task.image)
            else:
                print("Congratulations! All tasks are done!")
                return
        else:
            error = False
            try:
                ans, ans0 = labeler.label(task.image, task.texts)
            except NoValidAPIKey as e:
                print("All your API keys are expired.")
                exit()
            except Exception as e:
                error = True
                tasks.report_result(flag, "error")
                print('error', e)
            
            if not error:
                # js.append(ans0)
                print(task.comment, ans)
                print(score(ans))
                tasks.report_result(flag, ans)

thr = [Thread(target=work) for i in range(thr_cnt)]
for i in thr:
    i.start()
for i in thr:
    i.join()
# json.dump(js, open('log.json', 'w'))

# print(tasks.result)