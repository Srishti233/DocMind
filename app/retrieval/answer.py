"""Answer orchestration: rewrite -> cache -> route -> retrieve/SQL/summarise -> stream -> log.

``ask`` is a generator of event dicts:
  {"type": "status", "stage": ...}  progress
  {"type": "token", "text": ...}    answer text (streamed)
  {"type": "final", ...}            answer, sources, route, latencies
  {"type": "error", "message": ...} user-readable failure
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

from app import config, db
from app.ingest import spreadsheet
from app.llm import client as llm
from app.retrieval import memory, models, router, summarize, text2sql
from app.retrieval.cache import CacheEntry, semantic_cache
from app.retrieval.hybrid import Source, build_context, retrieve
from app.retrieval.injection import SYSTEM_PROMPT, wrap_source
from app.retrieval.store import get_store

log = logging.getLogger("docmind.answer")

NOT_FOUND = "I couldn't find this in the uploaded documents."
OUT_OF_SCOPE_MSG = ("That doesn't look like a question about your uploaded documents. "
                    "Ask about their contents, or switch workspace.")

Event = dict[str, Any]


@dataclass
class Result:
    """What a route handler produced."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    cacheable: bool = True


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _mentioned(question: str, doc: dict[str, Any]) -> bool:
    q = _norm(question)
    stem = _norm(doc["filename"].rsplit(".", 1)[0])
    return _norm(doc["filename"]) in q or (len(stem) >= 3 and stem in q)


def _resolve_docs(workspace: str, doc_type: str | None, filename: str | None) -> list[dict[str, Any]]:
    docs = db.list_docs(workspace, "done")
    if filename:
        docs = [d for d in docs if d["filename"] == filename]
    if doc_type:
        docs = [d for d in docs if d.get("doc_type") == doc_type]
    return docs


def _mark_cited(answer: str, sources: list[Source]) -> None:
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    for s in sources:
        s.cited = s.n in nums


def _stream_llm(prompt: str, system: str) -> Generator[Event, None, str]:
    parts: list[str] = []
    for piece in llm.stream(prompt, system):
        parts.append(piece)
        yield {"type": "token", "text": piece}
    return "".join(parts).strip()


def _is_not_found(answer: str) -> bool:
    return answer.strip().rstrip(".").lower().startswith(NOT_FOUND.rstrip(".").lower())


def _context_budget(question: str) -> int:
    s = config.settings
    return s.max_context_tokens - llm.estimate_tokens(SYSTEM_PROMPT) - llm.estimate_tokens(question) - 80


def _comparison_hits(q: str, ws: str, doc_type: str | None, filename: str | None, tl: dict[str, float]):
    s = config.settings
    docs = _resolve_docs(ws, doc_type, filename)
    mentioned = [d["filename"] for d in docs if _mentioned(q, d)]
    if len(mentioned) >= 2:
        targets = mentioned[:3]
    else:
        broad = retrieve(q, ws, doc_type=doc_type, filename=filename, top_k=s.candidates, timings=tl)
        targets = list(dict.fromkeys(h.payload.get("filename", "") for h in broad))[:3]
    hits = []
    for fn in targets:
        hits.extend(retrieve(q, ws, doc_type=doc_type, filename=fn, top_k=2, timings=tl))
    return hits


def _h_scope() -> Generator[Event, None, Result]:
    yield {"type": "token", "text": OUT_OF_SCOPE_MSG}
    return Result(OUT_OF_SCOPE_MSG, cacheable=False)


def _h_retrieval(q: str, ws: str, doc_type: str | None, filename: str | None, tl: dict[str, float],
                 comparison: bool = False) -> Generator[Event, None, Result]:
    s = config.settings
    yield {"type": "status", "stage": "retrieving"}
    t = time.perf_counter()
    if comparison:
        hits = _comparison_hits(q, ws, doc_type, filename, tl)
    else:
        hits = retrieve(q, ws, doc_type=doc_type, filename=filename, timings=tl)
    tl["retrieve_ms"] = (time.perf_counter() - t) * 1000
    ids, scores = [h.id for h in hits], [h.score for h in hits]
    if not hits or max(scores) < s.rerank_threshold:
        yield {"type": "token", "text": NOT_FOUND}
        return Result(NOT_FOUND, [], ids, scores, cacheable=False)
    context, sources = build_context(hits, _context_budget(q))
    task = ("Compare the documents. State which SOURCE supports each point, citing [n]."
            if comparison else "Answer using only the SOURCES and cite with [n].")
    yield {"type": "status", "stage": "generating"}
    t = time.perf_counter()
    answer = yield from _stream_llm(f"SOURCES:\n{context}\n\nQuestion: {q}\n\n{task}", SYSTEM_PROMPT)
    tl["llm_ms"] = (time.perf_counter() - t) * 1000
    if _is_not_found(answer):
        return Result(NOT_FOUND, [], ids, scores, cacheable=False)
    _mark_cited(answer, sources)
    return Result(answer, sources, ids, scores)


def _h_aggregation(q: str, tables: list[spreadsheet.TableInfo], tl: dict[str, float]
                   ) -> Generator[Event, None, Result]:
    yield {"type": "status", "stage": "querying spreadsheet"}
    t = time.perf_counter()
    try:
        sql, cols, rows, truncated = text2sql.query_tables(q, tables)
    except text2sql.SQLGenerationError as e:
        msg = f"I couldn't compute that from the spreadsheet: {e}"
        yield {"type": "token", "text": msg}
        return Result(msg, cacheable=False)
    tl["sql_ms"] = (time.perf_counter() - t) * 1000
    m = re.search(r"\b(?:from)\s+([a-z_][a-z0-9_]*)", sql, re.I)
    info = next((x for x in tables if m and x.table_name.lower() == m.group(1).lower()), tables[0])
    table_text = text2sql.format_result(cols, rows)
    src = Source(1, info.filename, None, f"SQL: {sql}", table_text[:300],
                 f"SQL: {sql}\n\n{table_text}"[:2200], info.doc_id, info.table_name, 0.0,
                 False, "spreadsheet", True)
    ids = [info.table_name]
    if not rows:
        ans = "The query returned no rows [1]."
    elif len(rows) == 1 and len(cols) == 1:
        v = rows[0][0]
        ans = f"{cols[0]} = {v:.6g} [1]." if isinstance(v, float) else f"{cols[0]} = {v} [1]."
    else:
        yield {"type": "status", "stage": "generating"}
        block = wrap_source(1, info.filename, None, f"SQL result ({len(rows)} rows)", table_text, False)
        t = time.perf_counter()
        ans = yield from _stream_llm(
            f"SOURCES:\n{block}\n\nQuestion: {q}\n\nAnswer from the SQL result only and cite [1]."
            + (" Note: results were truncated." if truncated else ""), SYSTEM_PROMPT)
        tl["llm_ms"] = (time.perf_counter() - t) * 1000
        return Result(ans, [src], ids, [])
    yield {"type": "token", "text": ans}
    return Result(ans, [src], ids, [])


def _h_summary(q: str, ws: str, doc_type: str | None, filename: str | None, tl: dict[str, float]
               ) -> Generator[Event, None, Result]:
    docs = _resolve_docs(ws, doc_type, filename)
    if len(docs) > 1:
        named = [d for d in docs if _mentioned(q, d)]
        if len(named) == 1:
            docs = named
    if not docs:
        msg = "There are no indexed documents to summarize here."
    elif len(docs) > 1:
        msg = ("Which document should I summarize? Mention its filename or pick one in the "
               "filename filter: " + ", ".join(d["filename"] for d in docs[:10]))
    else:
        msg = ""
    if msg:
        yield {"type": "token", "text": msg}
        return Result(msg, cacheable=False)
    doc = docs[0]
    chunks = get_store().iter_chunks(ws, doc["doc_id"])
    groups = summarize.plan_groups(chunks)
    if not groups:
        yield {"type": "token", "text": NOT_FOUND}
        return Result(NOT_FOUND, cacheable=False)
    sources = summarize.group_sources(groups)
    t = time.perf_counter()
    if len(groups) == 1:
        yield {"type": "status", "stage": "generating"}
        g = groups[0]
        block = wrap_source(1, g.filename, g.page_start, g.label, g.text, False)
        answer = yield from _stream_llm(
            f"SOURCES:\n{block}\n\nTask: {q}\nSummarize the document in 5 sentences or fewer, citing [1].",
            SYSTEM_PROMPT)
    else:
        partials = []
        for i, g in enumerate(groups, start=1):
            yield {"type": "status", "stage": f"summarizing section {i}/{len(groups)}"}
            partials.append(summarize.map_group(g, i))
        yield {"type": "status", "stage": "generating"}
        parts: list[str] = []
        for piece in summarize.reduce_stream(q, doc["filename"], partials):
            parts.append(piece)
            yield {"type": "token", "text": piece}
        answer = "".join(parts).strip()
    tl["llm_ms"] = (time.perf_counter() - t) * 1000
    _mark_cited(answer, sources)
    return Result(answer, sources, [g.chunk_id for g in groups], [])


def ask(question: str, workspace: str, *, doc_type: str | None = None, filename: str | None = None,
        session_id: str | None = None, use_cache: bool = True) -> Iterator[Event]:
    """Answer one question (see module docstring for the event protocol)."""
    q = (question or "").strip()
    ws = (workspace or "").strip()
    if not q:
        yield {"type": "error", "message": "Question is empty."}
        return
    if not ws:
        yield {"type": "error", "message": "workspace is required."}
        return
    tl: dict[str, float] = {}
    t0 = time.perf_counter()
    rewritten, route_name, reason = q, "unknown", ""
    vec: list[float] | None = None
    key = (ws, doc_type or "", filename or "")

    def finish(res: Result, cached: bool) -> Event:
        tl["total_ms"] = (time.perf_counter() - t0) * 1000
        src = [s.to_dict() if isinstance(s, Source) else s for s in res.sources]
        db.log_query(ws, q, rewritten, route_name, res.chunk_ids, res.scores, tl, res.answer, cached)
        memory.add_turn(session_id, rewritten, res.answer)
        return {"type": "final", "answer": res.answer, "sources": src, "route": route_name,
                "route_reason": reason, "rewritten": rewritten, "cached": cached,
                "latency_ms": {k: round(v, 1) for k, v in tl.items()}}

    try:
        t = time.perf_counter()
        rewritten = memory.rewrite_followup(q, memory.get_history(session_id))
        tl["rewrite_ms"] = (time.perf_counter() - t) * 1000

        if use_cache:
            t = time.perf_counter()
            vec = models.embed_query(rewritten)
            hit = semantic_cache.get(key, vec)
            tl["cache_ms"] = (time.perf_counter() - t) * 1000
            if hit:
                route_name, reason = hit.route, "semantic cache"
                yield {"type": "token", "text": hit.answer}
                yield finish(Result(hit.answer, list(hit.sources)), True)  # type: ignore[arg-type]
                return

        yield {"type": "status", "stage": "routing"}
        t = time.perf_counter()
        tables = [] if (doc_type and doc_type != "spreadsheet") else spreadsheet.list_tables(ws, filename)
        terms: set[str] = set()
        for tb in tables:
            terms |= set(re.findall(r"[a-z0-9]+", tb.table_name.lower()))
            terms |= {c for c, _ in tb.columns}
            terms |= set(re.findall(r"[a-z0-9]+", tb.filename.lower()))
        text_docs = [d for d in _resolve_docs(ws, doc_type, filename) if d.get("doc_type") != "spreadsheet"]
        decision = router.route_question(rewritten, has_tables=bool(tables), table_terms=terms,
                                         has_text_docs=bool(text_docs))
        route_name, reason = decision.name, f"{decision.reason} ({decision.source})"
        if route_name == "aggregation" and not tables:
            route_name, reason = "factual", "no spreadsheet tables available"
        tl["route_ms"] = (time.perf_counter() - t) * 1000
        yield {"type": "status", "stage": "routed", "route": route_name}

        if route_name == "out_of_scope":
            res = yield from _h_scope()
        elif route_name == "aggregation":
            res = yield from _h_aggregation(rewritten, tables, tl)
        elif route_name == "summary":
            res = yield from _h_summary(rewritten, ws, doc_type, filename, tl)
        else:
            res = yield from _h_retrieval(rewritten, ws, doc_type, filename, tl,
                                          comparison=(route_name == "comparison"))

        if res.cacheable and use_cache and vec is not None and res.answer != NOT_FOUND:
            semantic_cache.put(CacheEntry(key, vec, res.answer, [s.to_dict() for s in res.sources], route_name))
        yield finish(res, False)
    except llm.LLMError as e:
        _log_error(ws, q, rewritten, route_name, tl, t0, str(e))
        yield {"type": "error", "message": str(e)}
    except Exception as e:  # noqa: BLE001
        log.exception("ask failed")
        _log_error(ws, q, rewritten, route_name, tl, t0, f"{type(e).__name__}: {e}")
        yield {"type": "error", "message": f"Internal error: {type(e).__name__}: {e}"}


def _log_error(ws: str, q: str, rewritten: str, route: str, tl: dict[str, float], t0: float, err: str) -> None:
    tl["total_ms"] = (time.perf_counter() - t0) * 1000
    try:
        db.log_query(ws, q, rewritten, route, [], [], tl, "", False, err)
    except Exception:  # noqa: BLE001
        log.exception("could not log error")
