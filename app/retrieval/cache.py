"""In-memory semantic cache: cosine > threshold, capped size, LRU eviction."""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app import config


@dataclass
class CacheEntry:
    """A cached answer."""

    key: tuple[str, str, str]  # (workspace, doc_type, filename)
    vector: list[float]
    answer: str
    sources: list[dict[str, Any]]
    route: str


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class SemanticCache:
    """LRU semantic cache. Entries only match within the same (workspace, filters) key."""

    def __init__(self) -> None:
        self._d: OrderedDict[int, CacheEntry] = OrderedDict()
        self._next = 0
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, str], vector: list[float]) -> CacheEntry | None:
        """Best entry with cosine above the threshold; refreshes its LRU position."""
        thr = config.settings.cache_threshold
        with self._lock:
            best_id, best = None, thr
            for i, e in self._d.items():
                if e.key != key:
                    continue
                c = _cos(vector, e.vector)
                if c > best:
                    best_id, best = i, c
            if best_id is None:
                return None
            self._d.move_to_end(best_id)
            return self._d[best_id]

    def put(self, entry: CacheEntry) -> None:
        """Insert, evicting least-recently-used entries beyond CACHE_SIZE."""
        with self._lock:
            self._d[self._next] = entry
            self._next += 1
            while len(self._d) > config.settings.cache_size:
                self._d.popitem(last=False)

    def invalidate(self, workspace: str | None = None) -> None:
        """Drop entries for a workspace (call when its documents change)."""
        with self._lock:
            if workspace is None:
                self._d.clear()
            else:
                for i in [i for i, e in self._d.items() if e.key[0] == workspace]:
                    del self._d[i]

    def __len__(self) -> int:
        return len(self._d)


semantic_cache = SemanticCache()
