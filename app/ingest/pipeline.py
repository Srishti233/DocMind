"""Ingestion pipeline: register -> (queue) -> parse/classify/chunk/embed/upsert.

Memory profile: pages stream from the parser, chunkers are generators, and chunks are
embedded + upserted in batches of EMBED_BATCH, so a whole file's chunks/embeddings are
never held at once. Ingestion jobs run one at a time through a single worker thread.
"""
from __future__ import annotations

import hashlib
import logging
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from app import config, db
from app.ingest import classifier, metadata, spreadsheet
from app.ingest.chunkers import Chunk, chunk_document
from app.ingest.parsers import SPREADSHEET_EXTENSIONS, SUPPORTED_EXTENSIONS, ParseError, parse_pages, read_sample
from app.retrieval import models
from app.retrieval.cache import semantic_cache
from app.retrieval.store import Point, get_store

log = logging.getLogger("docmind.ingest")


@dataclass
class RegisterResult:
    """Outcome of registering an upload."""

    doc_id: str
    action: str  # "queued" | "replaced" | "duplicate"


def sanitize_filename(name: str) -> str:
    """Safe base filename (no directories, limited charset)."""
    base = Path(name.replace("\\", "/")).name.strip()
    base = re.sub(r"[^\w.\- ()]+", "_", base)[:150].strip(" .")
    if not base:
        raise ValueError("Invalid filename.")
    return base


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def remove_document(doc_id: str) -> bool:
    """Delete chunks, spreadsheet tables, the stored file and the registry row."""
    doc = db.get_doc(doc_id)
    if not doc:
        return False
    get_store().delete_by_doc(doc_id)
    spreadsheet.drop_tables(doc_id)
    try:
        Path(doc["path"]).unlink(missing_ok=True)
    except OSError:
        log.warning("could not remove file %s", doc["path"])
    db.delete_doc(doc_id)
    semantic_cache.invalidate(doc["workspace"])
    return True


def register_upload(workspace: str, filename: str, src: Path) -> RegisterResult:
    """Dedupe/replace logic. ``src`` is a temp file that is moved into the upload dir.

    * identical content already in the workspace -> "duplicate" (nothing re-indexed)
    * same filename, different content           -> old version removed first ("replaced")
    * otherwise                                  -> "queued"
    """
    s = config.settings
    safe = sanitize_filename(filename)
    ext = Path(safe).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        src.unlink(missing_ok=True)
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    size = src.stat().st_size
    if size == 0:
        src.unlink(missing_ok=True)
        raise ValueError("The file is empty.")
    if size > s.max_upload_bytes:
        src.unlink(missing_ok=True)
        raise ValueError(f"File is {size / 1048576:.1f} MB; the limit is {s.max_upload_mb} MB.")
    digest = sha256_file(src)
    dup = db.find_by_hash(workspace, digest)
    if dup and dup["status"] != "failed":
        src.unlink(missing_ok=True)
        return RegisterResult(dup["doc_id"], "duplicate")
    action = "queued"
    old = db.find_by_name(workspace, safe)
    if old:
        remove_document(old["doc_id"])
        action = "replaced"
    if dup and dup["status"] == "failed" and db.get_doc(dup["doc_id"]):
        remove_document(dup["doc_id"])
    doc_id = uuid.uuid4().hex[:12]
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = s.upload_dir / f"{doc_id}_{safe}"
    shutil.move(str(src), str(dest))
    db.insert_doc(doc_id, workspace, safe, digest, str(dest), size, "queued")
    semantic_cache.invalidate(workspace)
    return RegisterResult(doc_id, action)


def _alive(doc_id: str) -> bool:
    return db.get_doc(doc_id) is not None


def _make_points(doc: dict[str, Any], doc_type: str, chunks: list[Chunk], seq0: int,
                 extra: dict[str, Any] | None = None) -> list[Point]:
    """Embed a batch (with contextual prefix) and build Points."""
    texts = [f"[{doc['filename']} | {doc_type} | {c.section}]\n{c.text}" for c in chunks]
    dense = models.embed_dense(texts)
    sparse = models.embed_sparse(texts)
    now = time.time()
    pts = []
    for i, (c, d, (si, sv)) in enumerate(zip(chunks, dense, sparse)):
        payload = {
            "text": c.text, "page": c.page, "section": c.section, "parent_id": c.parent_id,
            "parent_text": c.parent_text, "file_hash": doc["file_hash"], "workspace": doc["workspace"],
            "doc_type": doc_type, "uploaded_at": now, "filename": doc["filename"],
            "doc_id": doc["doc_id"], "seq": seq0 + i,
        }
        payload.update(extra or {})
        pts.append(Point(str(uuid.uuid4()), d, si, sv, payload))
    return pts


def _index_stream(doc: dict[str, Any], doc_type: str, chunks: Iterator[Chunk]) -> int:
    """Embed + upsert in batches; abort if the document was deleted meanwhile."""
    batch_size = config.settings.embed_batch
    store = get_store()
    batch: list[Chunk] = []
    seq = 0

    def flush() -> None:
        nonlocal seq, batch
        if not batch:
            return
        if not _alive(doc["doc_id"]):
            raise ParseError("Document was deleted during ingestion.")
        store.upsert(_make_points(doc, doc_type, batch, seq))
        seq += len(batch)
        batch = []

    for c in chunks:
        batch.append(c)
        if len(batch) >= batch_size:
            flush()
    flush()
    return seq


def _ingest_text(doc: dict[str, Any]) -> None:
    path = Path(doc["path"])
    sample = read_sample(path)
    cls = classifier.classify(sample, doc["filename"])
    meta = metadata.extract_metadata(cls.doc_type, sample)
    db.update_doc(doc["doc_id"], doc_type=cls.doc_type)

    def stream() -> Iterator[Chunk]:
        mtext = metadata.metadata_chunk_text(cls.doc_type, doc["filename"], meta)
        if mtext:
            yield Chunk(mtext, 1 if path.suffix.lower() == ".pdf" else None, "Extracted metadata")
        yield from chunk_document(cls.doc_type, parse_pages(path))

    n = _index_stream(doc, cls.doc_type, stream())
    if n == 0 or (n == 1 and meta):
        raise ParseError("No text could be extracted from this file.")
    db.update_doc(doc["doc_id"], n_chunks=n, metadata=meta, status="done", error=None)


def _ingest_spreadsheet(doc: dict[str, Any]) -> None:
    infos = spreadsheet.load_file(doc)
    chunks = [Chunk(spreadsheet.schema_text(i), None, f"Table {i.table_name}") for i in infos]
    store = get_store()
    pts: list[Point] = []
    for i, (info, ch) in enumerate(zip(infos, chunks)):
        pts.extend(_make_points(doc, "spreadsheet", [ch], i, {"table_name": info.table_name}))
    if not _alive(doc["doc_id"]):
        raise ParseError("Document was deleted during ingestion.")
    store.upsert(pts)
    db.update_doc(doc["doc_id"], doc_type="spreadsheet", n_chunks=len(pts),
                  metadata={"tables": [{"name": i.table_name, "rows": i.row_count,
                                        "columns": [c for c, _ in i.columns]} for i in infos]},
                  status="done", error=None)


def ingest_document(doc_id: str) -> None:
    """Run the whole pipeline for one registered document (never raises)."""
    doc = db.get_doc(doc_id)
    if not doc:
        return
    db.update_doc(doc_id, status="processing", error=None)
    try:
        get_store().delete_by_doc(doc_id)  # idempotent restart
        spreadsheet.drop_tables(doc_id)
        if Path(doc["path"]).suffix.lower() in SPREADSHEET_EXTENSIONS:
            _ingest_spreadsheet(doc)
        else:
            _ingest_text(doc)
    except Exception as e:  # noqa: BLE001 - report every failure through the status endpoint
        log.exception("ingestion failed for %s", doc_id)
        msg = str(e) if isinstance(e, ParseError) else f"{type(e).__name__}: {e}"
        try:
            get_store().delete_by_doc(doc_id)
            spreadsheet.drop_tables(doc_id)
        except Exception:  # noqa: BLE001
            pass
        if _alive(doc_id):
            db.update_doc(doc_id, status="failed", error=msg[:500])
    finally:
        semantic_cache.invalidate(doc["workspace"])


class IngestQueue:
    """Single-worker FIFO: ingestion jobs run strictly one at a time."""

    def __init__(self) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the worker thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="docmind-ingest", daemon=True)
        self._thread.start()

    def submit(self, doc_id: str) -> None:
        """Enqueue a document id."""
        self._q.put(doc_id)

    def wait_idle(self, timeout: float | None = None) -> None:
        """Block until the queue drains (tests)."""
        end = None if timeout is None else time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if end is not None and time.monotonic() > end:
                raise TimeoutError("ingest queue did not drain")
            time.sleep(0.02)

    def stop(self) -> None:
        """Ask the worker to exit."""
        self._q.put(None)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                ingest_document(item)
            except Exception:  # noqa: BLE001
                log.exception("worker error")
            finally:
                self._q.task_done()

    def recover(self) -> int:
        """Re-queue documents left queued/processing by a previous run."""
        n = 0
        for d in db.list_docs():
            if d["status"] in {"queued", "processing"}:
                self.submit(d["doc_id"])
                n += 1
        return n


ingest_queue = IngestQueue()
