from collections import OrderedDict

class PredictionCache:

    def __init__(self, size=100):
        self.cache = OrderedDict()
        self.size = size

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        self.cache[key] = value

        if len(self.cache) > self.size:
            self.cache.popitem(last=False)