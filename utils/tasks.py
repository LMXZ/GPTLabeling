from threading import Semaphore
from .std import *
import time

def task_sign(image: str, text: str):
    return image + ',' + text

class TaskGroup:
    def __init__(self, images, texts) -> None:
        self.images = images
        self.texts = texts

class TaskUnit:
    def __init__(self, image, texts) -> None:
        self.image = image
        self.texts = texts
        self.status = 0
        self.time = 0.0
    
    def assign(self):
        self.status = 1
        self.time = time.time()
    
    def done(self):
        self.status = 2

class LabelingTask:
    def __init__(self, tasks: List[TaskGroup], result: Dict[str, int]) -> None:
        self.sema = Semaphore(1)
        self.tasks: List[TaskUnit] = []
        self.result = result
        for i in tasks:
            for j in i.images:
                tu = TaskUnit(j, i.texts)
                for k in i.texts:
                    if task_sign(j, k) in result.keys():
                        tu.status = 2
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

        self.tasks[p].status = 2
        for t, res in zip(self.tasks[p].texts, result):
            self.result[task_sign(self.tasks[p].image, t)] = res
        
        self.sema.release()
