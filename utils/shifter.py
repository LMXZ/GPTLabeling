from .std import *

class Shifter:
    def __init__(self, items: List) -> None:
        self.items = items
        self.p = 0
    
    def __call__(self):
        self.p %= len(self.items)
        res = self.items[self.p]
        self.p = (self.p + 1) % len(self.items)
        return res