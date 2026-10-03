"""Lazy singletons for the dense embedder, BM25 sparse model and cross-encoder.

Each model is loaded on first use, exactly once. ``DOCMIND_FAKE_MODELS=1`` (or
``set_overrides``) swaps in the hashing stand-ins from ``fakes`` (offline tests).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Iterable

from app import config
from app.retrieval import fakes

_lock = threading.Lock()
_dense: Any = None
_sparse: Any = None
_reranker: Any = None
_overrides: dict[str, Callable[..., Any]] = {}


def set_overrides(dense: Callable[..., Any] | None = None, sparse: Callable[..., Any] | None = None,
                  rerank: Callable[..., Any] | None = None) -> None:
    """Install replacement callables (tests). Call with no args to clear."""
    _overrides.clear()
    for k, v in (("dense", dense), ("sparse", sparse), ("rerank", rerank)):
        if v is not None:
            _overrides[k] = v


def use_fakes() -> None:
    """Install the hashing fakes as overrides."""
    set_overrides(dense=fakes.dense, sparse=fakes.sparse, rerank=fakes.rerank)


def _fake() -> bool:
    return config.settings.fake_models


def embedding_dim() -> int:
    """Dimension of the dense vectors."""
    if "dense" in _overrides or _fake():
        return fakes.DIM
    return config.settings.embed_dim


def _get_dense() -> Any:
    global _dense
    with _lock:
        if _dense is None:
            from fastembed import TextEmbedding
            _dense = TextEmbedding(model_name=config.settings.embed_model)
        return _dense


def _get_sparse() -> Any:
    global _sparse
    with _lock:
        if _sparse is None:
            from fastembed import SparseTextEmbedding
            _sparse = SparseTextEmbedding(model_name=config.settings.sparse_model)
        return _sparse


def _get_reranker() -> Any:
    global _reranker
    with _lock:
        if _reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _reranker = TextCrossEncoder(model_name=config.settings.rerank_model)
        return _reranker


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings in batches of EMBED_BATCH."""
    if "dense" in _overrides:
        return _overrides["dense"](texts)
    if _fake():
        return fakes.dense(texts)
    return [list(map(float, v)) for v in _get_dense().embed(texts, batch_size=config.settings.embed_batch)]


def embed_query(text: str) -> list[float]:
    """Dense embedding for a query (bge models use the same encoder for queries)."""
    return embed_dense([text])[0]


def _sparse_pairs(items: Iterable[Any]) -> list[tuple[list[int], list[float]]]:
    return [([int(i) for i in e.indices], [float(v) for v in e.values]) for e in items]


def embed_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """BM25 sparse vectors for documents."""
    if "sparse" in _overrides:
        return _overrides["sparse"](texts)
    if _fake():
        return fakes.sparse(texts)
    return _sparse_pairs(_get_sparse().embed(texts, batch_size=config.settings.embed_batch))


def embed_sparse_query(text: str) -> tuple[list[int], list[float]]:
    """BM25 sparse vector for a query."""
    if "sparse" in _overrides or _fake():
        return embed_sparse([text])[0]
    model = _get_sparse()
    gen = model.query_embed(text) if hasattr(model, "query_embed") else model.embed([text])
    return _sparse_pairs(gen)[0]


def rerank(query: str, docs: list[str]) -> list[float]:
    """Cross-encoder relevance scores (higher is better)."""
    if not docs:
        return []
    if "rerank" in _overrides:
        return list(_overrides["rerank"](query, docs))
    if _fake():
        return fakes.rerank(query, docs)
    return [float(x) for x in _get_reranker().rerank(query, docs)]


def loaded_models() -> dict[str, bool]:
    """Which heavy models are currently loaded in memory."""
    return {"embedder": _dense is not None, "sparse": _sparse is not None,
            "reranker": _reranker is not None}
