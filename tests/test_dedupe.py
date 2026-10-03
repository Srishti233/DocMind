"""Upload dedupe / replace / delete logic."""
import shutil
import tempfile
from pathlib import Path

import pytest

from app import config, db
from app.ingest import pipeline, spreadsheet
from app.retrieval.store import get_store
from tests.conftest import SAMPLES, ingest_file


def _tmp_copy(src: Path, text: str | None = None) -> Path:
    fd, tmp = tempfile.mkstemp(dir=config.settings.data_dir)
    Path(tmp).unlink()
    if text is None:
        shutil.copy(src, tmp)
    else:
        Path(tmp).write_text(text)
    return Path(tmp)


def _chunks_for(doc_id: str) -> int:
    d = db.get_doc(doc_id)
    return len(get_store().iter_chunks(d["workspace"], doc_id)) if d else 0


def test_identical_file_is_not_reindexed():
    d1 = ingest_file(SAMPLES / "hr_leave_policy.md")
    n_before = get_store().count()
    res = pipeline.register_upload("default", "hr_leave_policy.md", _tmp_copy(SAMPLES / "hr_leave_policy.md"))
    assert res.action == "duplicate" and res.doc_id == d1
    assert get_store().count() == n_before and len(db.list_docs("default")) == 1
    # identical content under a different name is also a duplicate
    res2 = pipeline.register_upload("default", "copy.md", _tmp_copy(SAMPLES / "hr_leave_policy.md"))
    assert res2.action == "duplicate" and res2.doc_id == d1


def test_same_content_in_another_workspace_is_independent():
    a = ingest_file(SAMPLES / "hr_leave_policy.md", "ws_a")
    b = ingest_file(SAMPLES / "hr_leave_policy.md", "ws_b")
    assert a != b and len(db.list_docs()) == 2


def test_changed_file_with_same_name_replaces_old_version():
    text_v1 = "# Policy\n\n## 1. Leave\nEmployees get 10 days of leave. This policy applies to all employees.\n"
    text_v2 = "# Policy\n\n## 1. Leave\nEmployees get 30 days of leave. This policy applies to all employees.\n"
    r1 = pipeline.register_upload("default", "p.md", _tmp_copy(None, text_v1))
    pipeline.ingest_document(r1.doc_id)
    old_path = Path(db.get_doc(r1.doc_id)["path"])
    assert _chunks_for(r1.doc_id) > 0 and old_path.exists()
    r2 = pipeline.register_upload("default", "p.md", _tmp_copy(None, text_v2))
    assert r2.action == "replaced" and r2.doc_id != r1.doc_id
    assert db.get_doc(r1.doc_id) is None and not old_path.exists()
    assert len(get_store().iter_chunks("default", r1.doc_id)) == 0  # old chunks deleted first
    pipeline.ingest_document(r2.doc_id)
    docs = db.list_docs("default")
    assert len(docs) == 1 and docs[0]["doc_id"] == r2.doc_id and docs[0]["status"] == "done"
    texts = " ".join(h.payload["text"] for h in get_store().iter_chunks("default", r2.doc_id))
    assert "30 days" in texts and "10 days" not in texts


def test_delete_removes_chunks_registry_and_file():
    d = ingest_file(SAMPLES / "service_agreement.txt")
    path = Path(db.get_doc(d)["path"])
    assert path.exists() and _chunks_for(d) > 0
    assert pipeline.remove_document(d) is True
    assert db.get_doc(d) is None and not path.exists() and get_store().count() == 0
    assert pipeline.remove_document(d) is False


def test_delete_spreadsheet_drops_tables():
    d = ingest_file(SAMPLES / "sales.csv")
    assert spreadsheet.list_tables("default")
    pipeline.remove_document(d)
    assert spreadsheet.list_tables("default") == []


def test_validation_errors():
    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.register_upload("default", "x.exe", _tmp_copy(None, "abc"))
    with pytest.raises(ValueError, match="empty"):
        pipeline.register_upload("default", "x.txt", _tmp_copy(None, ""))


def test_size_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")
    config.reload()
    with pytest.raises(ValueError, match="limit"):
        pipeline.register_upload("default", "x.txt", _tmp_copy(None, "abc"))


def test_filename_sanitised():
    assert pipeline.sanitize_filename("../../etc/passwd") == "passwd"
    assert pipeline.sanitize_filename("a b/c:d*.txt") == "c_d_.txt"


def test_failed_doc_can_be_reuploaded():
    r = pipeline.register_upload("default", "bad.pdf", _tmp_copy(None, "%PDF-garbage"))
    pipeline.ingest_document(r.doc_id)
    assert db.get_doc(r.doc_id)["status"] == "failed"
    r2 = pipeline.register_upload("default", "bad.pdf", _tmp_copy(None, "%PDF-garbage"))
    assert r2.action in {"queued", "replaced"} and r2.doc_id != r.doc_id
