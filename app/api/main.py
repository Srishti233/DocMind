"""FastAPI application: upload, status, delete, SSE chat, health, metrics, static UI."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app import config, db
from app.ingest import pipeline, spreadsheet
from app.ingest.classifier import DOC_TYPES
from app.ingest.parsers import SUPPORTED_EXTENSIONS
from app.llm import client as llm
from app.retrieval import answer, models

INDEX_HTML = Path(__file__).resolve().parents[2] / "static" / "index.html"
_ask_sem: asyncio.Semaphore | None = None  # one chat request at a time (created in lifespan)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create dirs/DBs, start the single ingest worker, resume interrupted jobs."""
    global _ask_sem
    config.ensure_dirs()
    db.init_db()
    spreadsheet.init_sheets_db()
    _ask_sem = asyncio.Semaphore(1)
    pipeline.ingest_queue.start()
    pipeline.ingest_queue.recover()
    yield
    pipeline.ingest_queue.stop()


app = FastAPI(title="DocMind", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    """Body of POST /ask."""

    question: str = Field(min_length=1, max_length=2000)
    workspace: str | None = None
    doc_type: str | None = None
    filename: str | None = None
    session_id: str | None = None
    use_cache: bool = True


def _doc_view(d: dict[str, Any]) -> dict[str, Any]:
    return {k: d.get(k) for k in ("doc_id", "filename", "workspace", "doc_type", "status", "error",
                                   "n_chunks", "size_bytes", "uploaded_at", "metadata")}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page UI."""
    if not INDEX_HTML.exists():
        raise HTTPException(500, "static/index.html is missing")
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    """Process RSS memory, loaded models, Ollama reachability."""
    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss
    ol = llm.check_ollama()
    msg = None
    if not ol["reachable"]:
        msg = "Ollama is not reachable. Start it with `ollama serve`."
    elif not ol["model_available"]:
        msg = f"Model not pulled. Run: ollama pull {ol['model']}"
    return {"status": "ok", "rss_mb": round(rss / 1048576, 1), "models_loaded": models.loaded_models(),
            "ollama": ol, "message": msg, "python": sys.version.split()[0]}


@app.get("/config")
def public_config() -> dict[str, Any]:
    """Values the UI needs."""
    s = config.settings
    return {"default_workspace": s.default_workspace, "doc_types": DOC_TYPES, "max_upload_mb": s.max_upload_mb,
            "extensions": sorted(SUPPORTED_EXTENSIONS)}


@app.get("/workspaces")
def workspaces() -> list[str]:
    """Known workspaces (always includes the default one)."""
    names = set(db.list_workspaces()) | {config.settings.default_workspace}
    return sorted(names)


@app.post("/documents", status_code=202)
async def upload(background: BackgroundTasks, file: UploadFile = File(...),
                 workspace: str = Form(config.settings.default_workspace)) -> dict[str, Any]:
    """Upload a file; ingestion runs in the background (poll /documents/{id}/status)."""
    ws = workspace.strip()
    if not ws or len(ws) > 64:
        raise HTTPException(400, "Invalid workspace name.")
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    s = config.settings
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    tmp = s.upload_dir / f"_tmp_{uuid.uuid4().hex}"
    size = 0
    try:
        with open(tmp, "wb") as out:  # streamed to disk: never hold the file in RAM
            while block := await file.read(1 << 20):
                size += len(block)
                if size > s.max_upload_bytes:
                    raise HTTPException(413, f"File exceeds the {s.max_upload_mb} MB upload limit.")
                out.write(block)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        res = await run_in_threadpool(pipeline.register_upload, ws, name, tmp)
    except ValueError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e)) from e
    if res.action != "duplicate":
        background.add_task(pipeline.ingest_queue.submit, res.doc_id)
    doc = db.get_doc(res.doc_id)
    return {"doc_id": res.doc_id, "action": res.action, "duplicate": res.action == "duplicate",
            "status": doc["status"] if doc else "unknown", "filename": doc["filename"] if doc else name}


@app.get("/documents")
def list_documents(workspace: str = Query(...)) -> list[dict[str, Any]]:
    """Documents in a workspace."""
    return [_doc_view(d) for d in db.list_docs(workspace)]


@app.get("/documents/{doc_id}/status")
def document_status(doc_id: str) -> dict[str, Any]:
    """queued | processing | done | failed (with error message)."""
    d = db.get_doc(doc_id)
    if not d:
        raise HTTPException(404, "Unknown document.")
    return _doc_view(d)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    """Remove chunks, tables, registry row and file."""
    ok = pipeline.remove_document(doc_id)
    if not ok:
        raise HTTPException(404, "Unknown document.")
    return {"deleted": doc_id}


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """Answer a question as Server-Sent Events: status*, token*, then final (or error)."""
    ws = (req.workspace or config.settings.default_workspace).strip()
    if req.doc_type and req.doc_type not in DOC_TYPES:
        raise HTTPException(400, f"doc_type must be one of {DOC_TYPES}")
    assert _ask_sem is not None, "app not started"

    async def events() -> AsyncIterator[str]:
        async with _ask_sem:  # only one LLM-backed request at a time
            gen = answer.ask(req.question, ws, doc_type=req.doc_type, filename=req.filename,
                             session_id=req.session_id, use_cache=req.use_cache)
            async for ev in iterate_in_threadpool(gen):
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/metrics/recent")
def metrics_recent(limit: int = Query(20, ge=1, le=200), workspace: str | None = None) -> list[dict[str, Any]]:
    """Recent query log rows (question, rewrite, route, chunk ids, scores, latencies, answer)."""
    return db.recent_queries(limit, workspace)
