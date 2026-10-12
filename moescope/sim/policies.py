"""Expert-cache replacement policies. A key is one (layer, expert) pair; a slot holds one expert.

Every cache starts cold except PinnedCache, which is preloaded and never changes.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable


class LRUCache:
    """Evict the least recently used expert."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[int, None] = OrderedDict()

    def access(self, key: int) -> bool:
        if key in self._items:
            self._items.move_to_end(key)
            return True
        if self.capacity > 0:
            if len(self._items) >= self.capacity:
                self._items.popitem(last=False)
            self._items[key] = None
        return False


class LFUCache:
    """Evict the expert with the fewest uses since it was admitted; ties go to the least recently used.

    O(1) per access: one LRU-ordered bucket per use count.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._freq: dict[int, int] = {}
        self._buckets: dict[int, OrderedDict[int, None]] = {}
        self._min = 0

    def _bump(self, key: int) -> None:
        f = self._freq[key]
        bucket = self._buckets[f]
        del bucket[key]
        if not bucket:
            del self._buckets[f]
            if self._min == f:
                self._min = f + 1
        self._freq[key] = f + 1
        self._buckets.setdefault(f + 1, OrderedDict())[key] = None

    def access(self, key: int) -> bool:
        if key in self._freq:
            self._bump(key)
            return True
        if self.capacity > 0:
            if len(self._freq) >= self.capacity:
                bucket = self._buckets[self._min]
                victim, _ = bucket.popitem(last=False)
                if not bucket:
                    del self._buckets[self._min]
                del self._freq[victim]
            self._freq[key] = 1
            self._buckets.setdefault(1, OrderedDict())[key] = None
            self._min = 1
        return False


class PinnedCache:
    """A fixed set of experts, chosen up front and never evicted; everything else streams in."""

    def __init__(self, pinned: Iterable[int]):
        self.pinned = frozenset(pinned)

    @classmethod
    def from_trace(cls, capacity: int, keys: Iterable[int]) -> PinnedCache:
        """Pin the `capacity` most-used keys (ties go to the key seen first)."""
        return cls(k for k, _ in Counter(keys).most_common(capacity))

    def access(self, key: int) -> bool:
        return key in self.pinned


def count_hits(cache, keys: Iterable[int]) -> int:
    access = cache.access
    return sum(1 for k in keys if access(k))


POLICIES = {"lru": LRUCache, "lfu": LFUCache}
