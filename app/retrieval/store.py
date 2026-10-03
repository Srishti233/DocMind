"""Vector store layer: embedded Qdrant (default) and an in-memory twin (tests/smoke).

Both implement the same interface. Every search REQUIRES a workspace (mandatory
filter); optional filters: doc_type, filename, doc_id(s).
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger("docmind.store")


@dataclass
class Point:
    """A chunk ready for upsert."""

    id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict[str, Any]


@dataclass
class Hit:
    """A search result."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Interface implemented by QdrantStore and MemoryStore."""

    def upsert(self, points: list[Point]) -> None: ...
    def delete_by_doc(self, doc_id: str) -> None: ...
    def search_dense(self, vector: list[float], workspace: str, limit: int, **f: Any) -> list[Hit]: ...
    def search_sparse(self, indices: list[int], values: list[float], workspace: str, limit: int,
                      **f: Any) -> list[Hit]: ...
    def iter_chunks(self, workspace: str, doc_id: str) -> list[Hit]: ...
    def count(self) -> int: ...


def _require_ws(workspace: str) -> None:
    if not workspace or not str(workspace).strip():
        raise ValueError("workspace is mandatory on every query")


def _match(payload: dict[str, Any], workspace: str, doc_type: str | None = None,
           filename: str | None = None, doc_id: str | None = None,
           doc_ids: list[str] | None = None) -> bool:
    if payload.get("workspace") != workspace:
        return False
    if doc_type and payload.get("doc_type") != doc_type:
        return False
    if filename and payload.get("filename") != filename:
        return False
    if doc_id and payload.get("doc_id") != doc_id:
        return False
    if doc_ids and payload.get("doc_id") not in doc_ids:
        return False
    return True


class MemoryStore:
    """Brute-force in-memory store with identical semantics (used by tests)."""

    def __init__(self) -> None:
        self._pts: dict[str, Point] = {}
        self._lock = threading.RLock()

    def upsert(self, points: list[Point]) -> None:
        with self._lock:
            for p in points:
                self._pts[p.id] = p

    def delete_by_doc(self, doc_id: str) -> None:
        with self._lock:
            for k in [k for k, p in self._pts.items() if p.payload.get("doc_id") == doc_id]:
                del self._pts[k]

    def search_dense(self, vector: list[float], workspace: str, limit: int, **f: Any) -> list[Hit]:
        _require_ws(workspace)
        with self._lock:
            pts = [p for p in self._pts.values() if _match(p.payload, workspace, **f)]
        hits = []
        for p in pts:
            dot = sum(a * b for a, b in zip(vector, p.dense))
            na = math.sqrt(sum(a * a for a in vector)) or 1.0
            nb = math.sqrt(sum(b * b for b in p.dense)) or 1.0
            hits.append(Hit(p.id, dot / (na * nb), p.payload))
        return sorted(hits, key=lambda h: -h.score)[:limit]

    def search_sparse(self, indices: list[int], values: list[float], workspace: str, limit: int,
                      **f: Any) -> list[Hit]:
        _require_ws(workspace)
        q = dict(zip(indices, values))
        with self._lock:
            pts = [p for p in self._pts.values() if _match(p.payload, workspace, **f)]
        hits = []
        for p in pts:
            d = dict(zip(p.sparse_indices, p.sparse_values))
            s = sum(v * d[i] for i, v in q.items() if i in d)
            if s > 0:
                hits.append(Hit(p.id, s, p.payload))
        return sorted(hits, key=lambda h: -h.score)[:limit]

    def iter_chunks(self, workspace: str, doc_id: str) -> list[Hit]:
        _require_ws(workspace)
        with self._lock:
            hits = [Hit(p.id, 0.0, p.payload) for p in self._pts.values()
                    if _match(p.payload, workspace, doc_id=doc_id)]
        return sorted(hits, key=lambda h: h.payload.get("seq", 0))

    def count(self) -> int:
        return len(self._pts)


class QdrantStore:
    """Qdrant in embedded local mode (path on disk, no server)."""

    def __init__(self, path: str, collection: str, dim: int) -> None:
        from qdrant_client import QdrantClient, models as qm

        self._qm = qm
        self._collection = collection
        self._lock = threading.RLock()
        self._client = QdrantClient(path=path)
        if not self._client.collection_exists(collection):
            # fastembed's BM25 emits term frequencies only; Qdrant must apply IDF server-side.
            self._client.create_collection(
                collection_name=collection,
                vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
        else:
            sparse = self._client.get_collection(collection).config.params.sparse_vectors or {}
            if getattr(sparse.get("sparse"), "modifier", None) != qm.Modifier.IDF:
                log.warning("The index in %s was built without the BM25 IDF modifier, so keyword "
                            "search quality is degraded. Delete that folder and re-upload your "
                            "documents.", path)

    def _filter(self, workspace: str, doc_type: str | None = None, filename: str | None = None,
                doc_id: str | None = None, doc_ids: list[str] | None = None) -> Any:
        qm = self._qm
        must = [qm.FieldCondition(key="workspace", match=qm.MatchValue(value=workspace))]
        if doc_type:
            must.append(qm.FieldCondition(key="doc_type", match=qm.MatchValue(value=doc_type)))
        if filename:
            must.append(qm.FieldCondition(key="filename", match=qm.MatchValue(value=filename)))
        if doc_id:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
        if doc_ids:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=list(doc_ids))))
        return qm.Filter(must=must)

    def upsert(self, points: list[Point]) -> None:
        qm = self._qm
        pts = [qm.PointStruct(
            id=p.id,
            vector={"dense": p.dense,
                    "sparse": qm.SparseVector(indices=p.sparse_indices, values=p.sparse_values)},
            payload=p.payload) for p in points]
        with self._lock:
            self._client.upsert(collection_name=self._collection, points=pts, wait=True)

    def delete_by_doc(self, doc_id: str) -> None:
        qm = self._qm
        with self._lock:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])),
                wait=True)

    def _query(self, query: Any, using: str, workspace: str, limit: int, **f: Any) -> list[Hit]:
        _require_ws(workspace)
        with self._lock:
            res = self._client.query_points(
                collection_name=self._collection, query=query, using=using,
                query_filter=self._filter(workspace, **f), limit=limit, with_payload=True)
        return [Hit(str(p.id), float(p.score), dict(p.payload or {})) for p in res.points]

    def search_dense(self, vector: list[float], workspace: str, limit: int, **f: Any) -> list[Hit]:
        return self._query(vector, "dense", workspace, limit, **f)

    def search_sparse(self, indices: list[int], values: list[float], workspace: str, limit: int,
                      **f: Any) -> list[Hit]:
        if not indices:
            return []
        return self._query(self._qm.SparseVector(indices=indices, values=values), "sparse",
                           workspace, limit, **f)

    def iter_chunks(self, workspace: str, doc_id: str) -> list[Hit]:
        _require_ws(workspace)
        hits: list[Hit] = []
        offset = None
        keys = ["text", "seq", "page", "section", "filename", "doc_id"]
        while True:
            with self._lock:
                pts, offset = self._client.scroll(
                    collection_name=self._collection, scroll_filter=self._filter(workspace, doc_id=doc_id),
                    limit=256, offset=offset, with_payload=keys, with_vectors=False)
            hits.extend(Hit(str(p.id), 0.0, dict(p.payload or {})) for p in pts)
            if offset is None:
                break
        return sorted(hits, key=lambda h: h.payload.get("seq", 0))

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection, exact=True).count)


_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_store() -> VectorStore:
    """Process-wide store singleton (created lazily from settings)."""
    global _store
    with _store_lock:
        if _store is None:
            from app import config
            from app.retrieval import models
            s = config.settings
            if s.store_backend == "memory":
                _store = MemoryStore()
            else:
                s.qdrant_path.mkdir(parents=True, exist_ok=True)
                _store = QdrantStore(str(s.qdrant_path), s.collection, models.embedding_dim())
        return _store


def set_store(store: VectorStore | None) -> None:
    """Install a store (tests); None resets so the next get_store() rebuilds it."""
    global _store
    _store = store
