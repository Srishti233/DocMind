"""Text-to-SQL safety validator and the read-only executor."""
import sqlite3
from pathlib import Path

import pytest

from app import config, db
from app.ingest import spreadsheet
from app.retrieval import text2sql
from app.retrieval.text2sql import SQLGenerationError, UnsafeSQL, run_query, validate_sql

SALES = Path(__file__).resolve().parent.parent / "samples" / "sales.csv"
ALLOWED = {"sales"}


def _load_sales():
    doc = {"doc_id": "d1", "workspace": "w", "filename": "sales.csv", "path": str(SALES)}
    return spreadsheet.load_file(doc)


def test_accepts_simple_selects():
    for q in ["SELECT region, SUM(units_sold) FROM sales GROUP BY region",
              "select avg(unit_price) from sales where region='N;orth';",
              "SELECT * FROM sales ORDER BY units_sold DESC LIMIT 3"]:
        assert validate_sql(q, ALLOWED)


def test_rejects_drop_update_delete_insert():
    for q in ["DROP TABLE sales", "UPDATE sales SET units_sold=1", "DELETE FROM sales",
              "INSERT INTO sales VALUES (1)", "ALTER TABLE sales ADD x", "CREATE TABLE t(x)",
              "PRAGMA table_info(sales)", "ATTACH DATABASE 'x' AS y", "REPLACE INTO sales VALUES (1)"]:
        with pytest.raises(UnsafeSQL):
            validate_sql(q, ALLOWED)


def test_rejects_multi_statement():
    for q in ["SELECT 1 FROM sales; DROP TABLE sales", "SELECT * FROM sales; SELECT * FROM sales",
              "SELECT * FROM sales;--"]:
        with pytest.raises(UnsafeSQL):
            validate_sql(q, ALLOWED)


def test_rejects_comments_quotes_subqueries_unions():
    for q in ["SELECT * FROM sales -- x", "SELECT * FROM sales /* x */",
              'SELECT "units_sold" FROM sales', "SELECT (SELECT 1) FROM sales",
              "SELECT * FROM sales UNION SELECT * FROM sales", "WITH x AS (SELECT 1) SELECT * FROM x"]:
        with pytest.raises(UnsafeSQL):
            validate_sql(q, ALLOWED)


def test_rejects_tables_outside_allow_list():
    for q in ["SELECT * FROM sqlite_master", "SELECT * FROM secrets",
              "SELECT * FROM sales, secrets", "SELECT * FROM sales JOIN secrets ON 1=1",
              "SELECT 1"]:
        with pytest.raises(UnsafeSQL):
            validate_sql(q, ALLOWED)


def test_semicolon_inside_string_literal_is_fine():
    assert validate_sql("SELECT * FROM sales WHERE region = 'a;b'", ALLOWED)


def test_run_query_executes_and_aggregates():
    _load_sales()
    cols, rows, trunc = run_query("SELECT region, SUM(units_sold) AS t FROM sales GROUP BY region ORDER BY region", ALLOWED)
    assert cols == ["region", "t"] and not trunc
    assert dict(rows) == {"East": 315, "North": 400, "South": 500, "West": 520}


def test_row_limit(monkeypatch):
    _load_sales()
    monkeypatch.setenv("SQL_ROW_LIMIT", "5")
    config.reload()
    cols, rows, trunc = run_query("SELECT * FROM sales", ALLOWED)
    assert len(rows) == 5 and trunc


def test_timeout(monkeypatch):
    _load_sales()
    monkeypatch.setenv("SQL_TIMEOUT_S", "0.05")
    config.reload()
    cross = " JOIN ".join(["sales"] * 8)  # 16**8 rows if it ever completed
    with pytest.raises(SQLGenerationError):
        run_query(f"SELECT COUNT(*) FROM {cross}", ALLOWED)


def test_connection_is_read_only_and_authorizer_blocks_other_tables():
    _load_sales()
    ro = sqlite3.connect(f"file:{config.settings.sheets_db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM sales")
    ro.close()
    # Table that exists in the DB but is not allow-listed: the authorizer layer denies it even
    # if the textual validator were bypassed.
    orig = text2sql.validate_sql
    text2sql.validate_sql = lambda sql, allowed: sql
    try:
        with pytest.raises(SQLGenerationError):
            run_query("SELECT * FROM sheet_tables", ALLOWED)
    finally:
        text2sql.validate_sql = orig


def test_extract_sql_from_fenced_output():
    assert text2sql.extract_sql("Sure!\n```sql\nSELECT 1 FROM sales;\n```\nDone") == "SELECT 1 FROM sales;"
    assert text2sql.extract_sql("SELECT a FROM sales").startswith("SELECT")
