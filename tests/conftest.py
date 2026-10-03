"""Shared fixtures: isolated data dir, in-memory store, fake models, stub LLM."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config, db  # noqa: E402
from app.ingest import spreadsheet  # noqa: E402
from app.llm import client as llm  # noqa: E402
from app.retrieval import memory, models, store  # noqa: E402
from app.retrieval.cache import semantic_cache  # noqa: E402
from tests.stubs import StubLLM  # noqa: E402

SAMPLES = ROOT / "samples"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Fresh data dir + offline fakes for every test (nothing touches the network)."""
    monkeypatch.setenv("DOCMIND_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STORE_BACKEND", "memory")
    for k in ("EXTRACT_METADATA", "RERANK_THRESHOLD", "SQL_ROW_LIMIT", "SQL_TIMEOUT_S",
              "MAX_UPLOAD_MB", "CACHE_SIZE", "MAX_PDF_PAGES"):
        monkeypatch.delenv(k, raising=False)
    config.reload()
    config.ensure_dirs()
    db.init_db()
    spreadsheet.init_sheets_db()
    st = store.MemoryStore()
    store.set_store(st)
    models.use_fakes()
    stub = StubLLM()
    llm.set_backend(stub)
    semantic_cache.invalidate()
    memory.clear()
    yield stub
    llm.set_backend(None)
    store.set_store(None)
    models.set_overrides()
    config.reload()


@pytest.fixture
def stub(env):
    """The active StubLLM."""
    return env


def ingest_file(src: Path, workspace: str = "default", name: str | None = None) -> str:
    """Copy a file to a temp location, register it and run the pipeline synchronously."""
    import shutil
    import tempfile

    from app.ingest import pipeline

    fd, tmp = tempfile.mkstemp(dir=config.settings.data_dir)
    Path(tmp).unlink()
    shutil.copy(src, tmp)
    res = pipeline.register_upload(workspace, name or src.name, Path(tmp))
    if res.action != "duplicate":
        pipeline.ingest_document(res.doc_id)
    return res.doc_id
