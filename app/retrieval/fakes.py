"""Deterministic hashing-based stand-ins for the embedding/sparse/rerank models.

Used by the offline test-suite and by ``DOCMIND_FAKE_MODELS=1`` smoke runs. They
need no downloads but are NOT semantically meaningful: do not use for real work.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 256
_STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "what", "how", "does",
         "do", "for", "on", "at", "by", "with", "it", "its", "be", "as", "this", "that", "i", "me",
         "my", "who", "which", "when", "where", "can", "per", "from", "about"}


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _h(tok: str) -> int:
    return int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)


def dense(texts: list[str]) -> list[list[float]]:
    """Hashed bag-of-words embedding, L2-normalised."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for tok in tokens(t):
            h = _h(tok)
            v[h % DIM] += 1.0 if (h >> 20) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """Hashed term-frequency sparse vectors."""
    out = []
    for t in texts:
        counts: dict[int, int] = {}
        for tok in tokens(t):
            i = _h(tok) % (2**31)
            counts[i] = counts.get(i, 0) + 1
        idx = sorted(counts)
        out.append((idx, [1.0 + math.log(counts[i]) for i in idx]))
    return out


def rerank(query: str, docs: list[str]) -> list[float]:
    """Score = query-token coverage mapped to [-5, 5] (mimics cross-encoder logits)."""
    q = set(tokens(query))
    scores = []
    for d in docs:
        dt = set(tokens(d))
        scores.append(-5.0 + 10.0 * (len(q & dt) / len(q) if q else 0.0))
    return scores
