from collections import OrderedDict, Counter
from typing import Optional

TIER_HOT = 0
TIER_COLD = 3

class LRUCache:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._cache: OrderedDict = OrderedDict()

    def access(self, block_id: int) -> int:
        if block_id in self._cache:
            self._cache.move_to_end(block_id)
            return TIER_HOT
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[block_id] = True
        return TIER_COLD

    def reset(self):
        self._cache.clear()

class LFUCache:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self._cache: set = set()
        self._freq: Counter = Counter()

    def access(self, block_id: int) -> int:
        if block_id in self._cache:
            self._freq[block_id] += 1
            return TIER_HOT
        if len(self._cache) >= self.capacity:
            victim = min(self._cache, key=lambda bid: self._freq[bid])
            self._cache.discard(victim)
            del self._freq[victim]
        self._cache.add(block_id)
        self._freq[block_id] = 1
        return TIER_COLD

    def reset(self):
        self._cache.clear()
        self._freq.clear()

class StaticCache:
    def access(self, block_id: int) -> int:
        return TIER_COLD

    def reset(self):
        pass
