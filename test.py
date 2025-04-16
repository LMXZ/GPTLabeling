from utils.std import *
from utils.config import *
import requests
import datetime
from labelers.gpt_labeler import GPTLabeler
from utils.tasks import *
import threading
from threading import Thread

data = json.load(open('test_data/reid_raw.json'))

groups: List[TaskGroup] = []
for i in data:
    id = i['id'] - 1
    while len(groups) - 1 < id:
        groups.append(TaskGroup([], []))
    groups[id].images.append(os.path.join(config['data_root'], i['file_path']))
    for j in i['captions']:
        groups[id].texts.append(j)

tasks = LabelingTask(groups, dict())

js = []

def work():
    labeler = GPTLabeler()
    while True:
        flag, task = tasks.get_task()
        if flag == -1:
            if task.image is not None:
                time.sleep(task.image)
            else:
                return
        else:
            try:
                ans, ans0 = labeler.label(task.image, task.texts)
            except:
                print("error")
            js.append(ans0)
            print(ans)
            tasks.report_result(flag, ans)


thr = Thread(target=work)
thr.start()
thr.join()
json.dump(js, open('log.json', 'w'))

print(tasks.result)