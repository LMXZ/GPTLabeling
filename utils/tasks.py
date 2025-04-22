from threading import Semaphore
from .std import *
import time
from .lmdb_dict import LmdbDict
from hashlib import sha256

def task_sign(image: str, text: str):
    z = image + ',' + text
    # print()
    # print(z)
    res = sha256((z).encode('utf8')).hexdigest()
    # print(res)
    # exit()
    return res

class TaskGroup:
    def __init__(self, images, texts, image_ids, text_ids) -> None:
        self.images = images
        self.texts = texts
        self.image_ids = image_ids
        self.text_ids = text_ids

class TaskUnit:
    def __init__(self, image, texts, comment='') -> None:
        self.image = image
        self.texts = texts
        self.status = 0
        self.time = 0.0
        self.comment = comment
    
    def assign(self):
        self.status = 1
        self.time = time.time()
    
    def done(self):
        self.status = 2

class LabelingTask:
    def __init__(self, tasks: List[TaskGroup], result: str) -> None:
        self.sema = Semaphore(1)
        self.tasks: List[TaskUnit] = []
        result = LmdbDict(result)
        self.result = result
        keys = result.keys()
        for i in tqdm(tasks):
            for j, j_id in zip(i.images, i.image_ids):
                tu = TaskUnit(j, i.texts)
                tu.comment = str(j_id)
                tu.status = 2
                for k, k_ids in zip(i.texts, i.text_ids):
                    if not task_sign(j, k) in keys:
                        tu.status = 0
                        break
                self.tasks.append(tu)
        self.p = 0

    def get_task(self):
        self.sema.acquire()

        res = -1, TaskUnit(None, None)
        done_cnt = 0
        wait_time = 99999999999999999
        for i in range(len(self.tasks)):
            if self.tasks[self.p].status == 2:
                done_cnt += 1
            elif self.tasks[self.p].status == 1:
                wait_time = min(wait_time, self.tasks[self.p].time + 300 - time.time())
            if self.tasks[self.p].status == 0 or \
                (self.tasks[self.p].status == 1 and time.time() - self.tasks[self.p].time > 300):
                res = self.p, self.tasks[self.p]
                self.tasks[self.p].assign()
                self.p = (self.p + 1) % len(self.tasks)
                break
            self.p = (self.p + 1) % len(self.tasks)

        if done_cnt == len(self.tasks):
            res = -1, TaskUnit(None, None)
        elif res[0] == -1:
            res = -1, TaskUnit(wait_time + 0.1, wait_time + 0.1)
        
        self.sema.release()
        return res
    
    def report_result(self, p, result: List[int]):
        self.sema.acquire()

        if isinstance(result, str):
            if self.tasks[p].status == 1:
                self.tasks[p].status = 0
        else:
            try:
                if len(self.tasks[p].texts) != len(result):
                    raise Exception("Bad response!")
                for t, res in zip(self.tasks[p].texts, result):
                    self.result[task_sign(self.tasks[p].image, t)] = str(res)
                self.tasks[p].status = 2
            except Exception as e:
                print('db error: ', e)
                self.tasks[p].status = 0
        
        self.sema.release()
