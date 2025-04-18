from .std import *
from threading import Semaphore

class Shifter:
    def __init__(self, items: List) -> None:
        self.items = items.copy()
        self.p = 0
        self.sema = Semaphore(1)
    
    def __call__(self):
        self.sema.acquire()

        if len(self.items) == 0:
            res = -1, None
        else:
            self.p %= len(self.items)
            res = self.p, self.items[self.p]
            self.p = (self.p + 1) % len(self.items)

        self.sema.release()
        return res
    
    def __len__(self):
        return len(self.items)

    def remove(self, items_to_remove: List[int]):
        self.sema.acquire()

        items_to_remove = set(items_to_remove)
        for i in range(len(self)-1, -1, -1):
            if self.items[i] in items_to_remove:
                self.items.pop(i)
        
        self.sema.release()
