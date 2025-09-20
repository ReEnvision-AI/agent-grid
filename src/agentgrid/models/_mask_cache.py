from __future__ import annotations

from collections import OrderedDict
from typing import Hashable


class LRUMaskCache:
    """A small LRU cache for storing attention masks keyed by tensor metadata."""

    def __init__(self, max_size: int = 32):
        self.max_size = max_size
        self._cache: OrderedDict[Hashable, object] = OrderedDict()

    def get(self, key: Hashable):
        value = self._cache.pop(key, None)
        if value is not None:
            self._cache[key] = value
        return value

    def put(self, key: Hashable, value) -> None:
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
