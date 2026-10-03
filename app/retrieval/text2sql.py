"""Text-to-SQL over the spreadsheet SQLite DB with defence in depth.

Layers: (1) textual validator (one SELECT, no comments/quotes/keywords, allow-listed
tables), (2) SQLite authorizer callback, (3) read-only connection, (4) row limit,
(5) wall-clock timeout via progress handler.
"""
from __future__ import annotations

import re
import sqlite3
import time
from typing import Any

from app import config
from app.ingest.spreadsheet import TableInfo
from app.llm import client as llm


class UnsafeSQL(ValueError):
    """The SQL failed validation."""


class SQLGenerationError(RuntimeError):
    """Could not produce a working query."""


_FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "replace", "attach", "detach",
    "pragma", "vacuum", "reindex", "truncate", "begin", "commit", "rollback", "savepoint",
    "release", "load_extension", "with", "union", "intersect", "except", "returning",
}


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_sql(sql: str, allowed_tables: set[str]) -> str:
    """Return the cleaned statement or raise UnsafeSQL.

    Accepts exactly one SELECT (no subqueries/CTEs/unions), no comments, no quoted
    identifiers, and only allow-listed tables.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQL("Empty SQL.")
    cleaned = sql.strip().rstrip(";").strip()
    bare = _strip_literals(cleaned)
    if ";" in bare:
        raise UnsafeSQL("Multiple statements are not allowed.")
    if "--" in bare or "/*" in bare or "*/" in bare:
        raise UnsafeSQL("SQL comments are not allowed.")
    if any(ch in bare for ch in ('"', "`", "[", "]")):
        raise UnsafeSQL("Quoted identifiers are not allowed; use plain table/column names.")
    words = re.findall(r"[a-z_][a-z0-9_]*", bare.lower())
    if not words or words[0] != "select":
        raise UnsafeSQL("Only SELECT statements are allowed.")
    bad = _FORBIDDEN & set(words)
    if bad:
        raise UnsafeSQL(f"Forbidden keyword(s): {', '.join(sorted(bad))}.")
    if words.count("select") != 1:
        raise UnsafeSQL("Exactly one SELECT is allowed (no subqueries).")
    allowed = {t.lower() for t in allowed_tables}
    if "from" not in words:
        raise UnsafeSQL("A FROM clause with an allowed table is required.")
    for m in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", bare, re.I):
        if m.group(1).lower() not in allowed:
            raise UnsafeSQL(f"Table '{m.group(1)}' is not available.")
    # comma joins: FROM a, b
    fm = re.search(r"\bfrom\b(.*?)(?:\bwhere\b|\bgroup\b|\border\b|\blimit\b|\bhaving\b|$)", bare, re.I | re.S)
    if fm:
        for part in re.split(r",|\bjoin\b|\bon\b[^,]*?(?=\bjoin\b|$)", fm.group(1), flags=re.I):
            tok = part.strip().split()
            if tok and tok[0].lower() not in allowed and re.fullmatch(r"[a-z_][a-z0-9_]*", tok[0], re.I) \
                    and tok[0].lower() not in {"inner", "left", "right", "outer", "cross", "natural"}:
                raise UnsafeSQL(f"Table '{tok[0]}' is not available.")
    return cleaned


def run_query(sql: str, allowed_tables: set[str]) -> tuple[list[str], list[tuple[Any, ...]], bool]:
    """Validate and execute. Returns (columns, rows, truncated)."""
    s = config.settings
    sql = validate_sql(sql, allowed_tables)
    allowed = {t.lower() for t in allowed_tables}
    banned_funcs = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

    def authorizer(action: int, a1: str | None, a2: str | None, db: str | None, src: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            return sqlite3.SQLITE_OK if (a1 or "").lower() in allowed else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (a2 or "").lower() in banned_funcs else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    if not s.sheets_db.exists():
        raise SQLGenerationError("No spreadsheet data has been loaded.")
    con = sqlite3.connect(f"file:{s.sheets_db}?mode=ro", uri=True, timeout=5)
    try:
        con.execute("PRAGMA query_only=ON")
        deadline = time.monotonic() + s.sql_timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        con.set_authorizer(authorizer)
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(s.sql_row_limit + 1)
        except sqlite3.Error as e:
            raise SQLGenerationError(f"SQL failed: {e}") from e
        truncated = len(rows) > s.sql_row_limit
        return cols, [tuple(r) for r in rows[:s.sql_row_limit]], truncated
    finally:
        con.close()


def _schema_prompt(tables: list[TableInfo], question: str, max_tables: int = 4) -> str:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))

    def rel(t: TableInfo) -> int:
        names = {t.table_name} | {c for c, _ in t.columns} | set(re.findall(r"[a-z0-9]+", t.filename.lower()))
        return -len(words & names)

    parts = []
    for t in sorted(tables, key=rel)[:max_tables]:
        cols = ", ".join(f"{c} {ty}" for c, ty in t.columns)
        ex = "; ".join(", ".join(str(v) for v in row) for row in t.sample[:2])
        parts.append(f"TABLE {t.table_name} ({cols})  -- {t.row_count} rows; e.g. {ex[:300]}")
    return "\n".join(parts)


def extract_sql(text: str) -> str:
    """Pull a SELECT statement out of model output (fences, prose)."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        text = m.group(1)
    m = re.search(r"\bselect\b.*", text, re.S | re.I)
    return (m.group(0) if m else text).strip().split("\n\n")[0].strip()


def query_tables(question: str, tables: list[TableInfo]
                 ) -> tuple[str, list[str], list[tuple[Any, ...]], bool]:
    """Generate SQL with the local LLM, run it safely, retry once on error.

    Returns (sql, columns, rows, truncated). Raises SQLGenerationError.
    """
    allowed = {t.table_name for t in tables}
    schema = _schema_prompt(tables, question)
    system = ("You write ONE SQLite SELECT statement. Rules: a single SELECT only; no subqueries, "
              "CTEs, UNION or comments; do not quote identifiers; use only the tables and columns "
              "listed. Use aggregate functions and GROUP BY when asked. Output only the SQL.")
    prompt = f"{schema}\n\nQuestion: {question}\nSQL:"
    last_err = ""
    for attempt in range(2):
        try:
            raw = llm.generate(prompt if attempt == 0 else
                               f"{prompt}\n\nYour previous attempt was rejected: {last_err}\nWrite a corrected SQL.",
                               system=system, num_predict=200, temperature=0.0)
        except llm.LLMError:
            raise
        sql = extract_sql(raw)
        try:
            cols, rows, trunc = run_query(sql, allowed)
            return sql, cols, rows, trunc
        except (UnsafeSQL, SQLGenerationError) as e:
            last_err = str(e)
    raise SQLGenerationError(last_err or "Could not generate a valid query.")


def format_result(cols: list[str], rows: list[tuple[Any, ...]], max_rows: int = 20) -> str:
    """Plain-text table for prompts and citations."""
    lines = [" | ".join(cols)]
    for r in rows[:max_rows]:
        lines.append(" | ".join("" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)
