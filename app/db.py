"""SQLite registry (documents) and query log. Stdlib sqlite3 only."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app import config

_DOC_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    doc_type TEXT,
    status TEXT NOT NULL,
    error TEXT,
    n_chunks INTEGER DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    path TEXT NOT NULL,
    metadata TEXT,
    uploaded_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_ws_name ON documents(workspace, filename);
CREATE INDEX IF NOT EXISTS idx_docs_ws_hash ON documents(workspace, file_hash);
"""

_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    workspace TEXT,
    question TEXT,
    rewritten TEXT,
    route TEXT,
    chunk_ids TEXT,
    scores TEXT,
    latencies TEXT,
    answer TEXT,
    cached INTEGER DEFAULT 0,
    error TEXT
);
"""


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection that commits on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create registry and log tables if they do not exist."""
    s = config.settings
    with connect(s.registry_db) as con:
        con.executescript(_DOC_SCHEMA)
    with connect(s.logs_db) as con:
        con.executescript(_LOG_SCHEMA)


def _doc(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"]) if d.get("metadata") else {}
    return d


def insert_doc(doc_id: str, workspace: str, filename: str, file_hash: str, path: str,
               size_bytes: int, status: str = "queued") -> None:
    """Insert a new registry row."""
    now = time.time()
    with connect(config.settings.registry_db) as con:
        con.execute(
            "INSERT INTO documents (doc_id, workspace, filename, file_hash, status, path,"
            " size_bytes, uploaded_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, workspace, filename, file_hash, status, path, size_bytes, now, now),
        )


def update_doc(doc_id: str, **fields: Any) -> None:
    """Update arbitrary columns of a registry row."""
    if not fields:
        return
    if "metadata" in fields and not isinstance(fields["metadata"], str):
        fields["metadata"] = json.dumps(fields["metadata"])
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect(config.settings.registry_db) as con:
        con.execute(f"UPDATE documents SET {cols} WHERE doc_id=?", (*fields.values(), doc_id))


def get_doc(doc_id: str) -> dict[str, Any] | None:
    """Fetch one document row."""
    with connect(config.settings.registry_db) as con:
        return _doc(con.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone())


def find_by_name(workspace: str, filename: str) -> dict[str, Any] | None:
    """Find a document by (workspace, filename)."""
    with connect(config.settings.registry_db) as con:
        return _doc(con.execute(
            "SELECT * FROM documents WHERE workspace=? AND filename=?", (workspace, filename)
        ).fetchone())


def find_by_hash(workspace: str, file_hash: str) -> dict[str, Any] | None:
    """Find a document by (workspace, sha256)."""
    with connect(config.settings.registry_db) as con:
        return _doc(con.execute(
            "SELECT * FROM documents WHERE workspace=? AND file_hash=? ORDER BY uploaded_at DESC",
            (workspace, file_hash),
        ).fetchone())


def list_docs(workspace: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """List documents, optionally filtered by workspace and status."""
    q, args = "SELECT * FROM documents WHERE 1=1", []
    if workspace:
        q += " AND workspace=?"
        args.append(workspace)
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY uploaded_at DESC"
    with connect(config.settings.registry_db) as con:
        return [_doc(r) for r in con.execute(q, args).fetchall()]  # type: ignore[misc]


def list_workspaces() -> list[str]:
    """Distinct workspaces that contain documents."""
    with connect(config.settings.registry_db) as con:
        return [r[0] for r in con.execute("SELECT DISTINCT workspace FROM documents ORDER BY 1")]


def delete_doc(doc_id: str) -> None:
    """Delete a registry row."""
    with connect(config.settings.registry_db) as con:
        con.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))


def log_query(workspace: str, question: str, rewritten: str, route: str,
              chunk_ids: list[str], scores: list[float], latencies: dict[str, float],
              answer: str, cached: bool = False, error: str | None = None) -> None:
    """Persist one query record."""
    with connect(config.settings.logs_db) as con:
        con.execute(
            "INSERT INTO query_logs (ts, workspace, question, rewritten, route, chunk_ids,"
            " scores, latencies, answer, cached, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), workspace, question, rewritten, route, json.dumps(chunk_ids),
             json.dumps([round(float(x), 4) for x in scores]),
             json.dumps({k: round(v, 1) for k, v in latencies.items()}),
             answer, int(cached), error),
        )


def recent_queries(limit: int = 20, workspace: str | None = None) -> list[dict[str, Any]]:
    """Most recent query log rows, newest first."""
    q, args = "SELECT * FROM query_logs", []
    if workspace:
        q += " WHERE workspace=?"
        args.append(workspace)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(max(1, min(limit, 200)))
    with connect(config.settings.logs_db) as con:
        rows = con.execute(q, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("chunk_ids", "scores", "latencies"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        out.append(d)
    return out
