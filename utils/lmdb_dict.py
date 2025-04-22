from .std import *
import lmdb

class LmdbDict:
    def __init__(self, path) -> None:
        self.path = path
    
    def __getitem__(self, key):
        res = ''
        with lmdb.open(self.path, map_size=1099511627776) as env:
            with env.begin() as txn:
                res = txn.get(key.encode('utf8'))
        return res.decode('utf8')

    def __setitem__(self, key, value):
        with lmdb.open(self.path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                txn.put(key.encode('utf8'), value.encode('utf8'))
    
    def keys(self):
        res = set()
        with lmdb.open(self.path, map_size=1099511627776) as env:
            with env.begin() as txn:
                for key, value in txn.cursor():
                    res.add(key.decode('utf8'))
        return res
    
    def items(self):
        res = []
        with lmdb.open(self.path, map_size=1099511627776) as env:
            with env.begin() as txn:
                for key, value in txn.cursor():
                    res.append((key.decode('utf8'), value.decode('utf8')))
        return res

if __name__ == '__main__':
    db = LmdbDict('./scores_db')
    print(len(db.items()))
