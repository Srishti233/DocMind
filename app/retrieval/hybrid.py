"""Hybrid retrieval: dense + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app import config
from app.retrieval import models
from app.retrieval.injection import scan, wrap_source
from app.retrieval.store import Hit, get_store

Mode = Literal["dense", "hybrid", "hybrid_rerank"]
RRF_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. ``rankings`` are lists of ids, best first.

    score(id) = sum over rankings of 1 / (k + rank), rank starting at 1. Ties keep the
    order of first appearance, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking, start=1):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
    order = {_id: i for i, _id in enumerate(dict.fromkeys(i for r in rankings for i in r))}
    return sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))


def retrieve(query: str, workspace: str, *, doc_type: str | None = None, filename: str | None = None,
             doc_ids: list[str] | None = None, mode: Mode = "hybrid_rerank",
             candidates: int | None = None, top_k: int | None = None,
             timings: dict[str, float] | None = None) -> list[Hit]:
    """Search one workspace (mandatory) and return the best ``top_k`` hits.

    ``Hit.score`` is the cross-encoder score in ``hybrid_rerank`` mode, otherwise the
    fusion / cosine score.
    """
    if not workspace or not workspace.strip():
        raise ValueError("workspace is mandatory on every query")
    s = config.settings
    n_cand = candidates or s.candidates
    k = top_k or s.top_k
    filt: dict[str, Any] = {"doc_type": doc_type, "filename": filename, "doc_ids": doc_ids}
    store = get_store()
    tm = timings if timings is not None else {}

    t = time.perf_counter()
    dense_hits = store.search_dense(models.embed_query(query), workspace, n_cand, **filt)
    tm["dense_ms"] = tm.get("dense_ms", 0.0) + (time.perf_counter() - t) * 1000
    if mode == "dense":
        return dense_hits[:k]

    t = time.perf_counter()
    si, sv = models.embed_sparse_query(query)
    sparse_hits = store.search_sparse(si, sv, workspace, n_cand, **filt)
    by_id = {h.id: h for h in dense_hits + sparse_hits}
    fused = rrf_fuse([[h.id for h in dense_hits], [h.id for h in sparse_hits]])[:n_cand]
    fused_hits = [Hit(i, sc, by_id[i].payload) for i, sc in fused]
    tm["sparse_fuse_ms"] = tm.get("sparse_fuse_ms", 0.0) + (time.perf_counter() - t) * 1000
    if mode == "hybrid":
        return fused_hits[:k]

    t = time.perf_counter()
    docs = [f"{h.payload.get('section', '')}\n{h.payload.get('text', '')}" for h in fused_hits]
    scores = models.rerank(query, docs)
    ranked = sorted(zip(fused_hits, scores), key=lambda x: -x[1])[:k]
    tm["rerank_ms"] = tm.get("rerank_ms", 0.0) + (time.perf_counter() - t) * 1000
    return [Hit(h.id, float(sc), h.payload) for h, sc in ranked]


@dataclass
class Source:
    """A numbered source passage shown to the LLM and to the user."""

    n: int
    filename: str
    page: int | None
    section: str
    snippet: str
    passage: str
    doc_id: str
    chunk_id: str
    score: float
    flagged: bool = False
    doc_type: str = ""
    cited: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view for the API."""
        return {"n": self.n, "filename": self.filename, "page": self.page, "section": self.section,
                "snippet": self.snippet, "passage": self.passage, "doc_id": self.doc_id,
                "chunk_id": self.chunk_id, "score": round(self.score, 3), "flagged": self.flagged,
                "doc_type": self.doc_type, "cited": self.cited}


def build_context(hits: list[Hit], budget_tokens: int) -> tuple[str, list[Source]]:
    """Number passages, expand to parent sections, dedupe, and truncate to the token budget."""
    budget_chars = max(400, budget_tokens * 4)
    used = 0
    blocks: list[str] = []
    sources: list[Source] = []
    seen_parents: set[str] = set()
    for h in hits:
        p = h.payload
        pid = p.get("parent_id")
        if pid:
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
        passage = p.get("parent_text") or p.get("text") or ""
        remaining = budget_chars - used - 160  # header/footer overhead
        if remaining < 200:
            break
        if len(passage) > remaining:
            passage = passage[:remaining].rstrip() + " …"
        n = len(sources) + 1
        flagged = bool(scan(passage))
        block = wrap_source(n, p.get("filename", "?"), p.get("page"), p.get("section", ""), passage, flagged)
        used += len(block)
        blocks.append(block)
        sources.append(Source(n, p.get("filename", "?"), p.get("page"), p.get("section", ""),
                              (p.get("text") or "")[:300], passage[:2200], p.get("doc_id", ""), h.id,
                              h.score, flagged, p.get("doc_type", "")))
    return "\n\n".join(blocks), sources
