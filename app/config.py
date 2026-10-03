"""Central configuration. Every setting is an env var with a safe default.

Usage: ``from app import config`` then ``config.settings.<name>`` at call time
(never ``from app.config import settings``), so ``config.reload()`` works in tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from ``path`` into os.environ without overriding real variables."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of all runtime settings."""

    # storage
    data_dir: Path
    upload_dir: Path
    qdrant_path: Path
    registry_db: Path
    logs_db: Path
    sheets_db: Path
    collection: str
    default_workspace: str
    store_backend: str  # "qdrant" (default) or "memory" (tests / smoke runs)
    # llm
    ollama_url: str
    llm_model: str
    num_ctx: int
    num_predict: int
    temperature: float
    keep_alive: str
    llm_timeout_s: float
    # models
    embed_model: str
    embed_dim: int
    sparse_model: str
    rerank_model: str
    embed_batch: int
    fake_models: bool  # hashing-based stand-ins, for offline smoke tests only
    # limits
    max_upload_mb: int
    max_pdf_pages: int
    # retrieval
    candidates: int
    top_k: int
    child_chars: int
    parent_chars: int
    max_context_tokens: int
    rerank_threshold: float
    history_turns: int
    summary_max_sections: int
    # features
    extract_metadata: bool
    classifier_min_score: float
    classifier_margin: float
    # cache
    cache_size: int
    cache_threshold: float
    # sql
    sql_row_limit: int
    sql_timeout_s: float

    @property
    def max_upload_bytes(self) -> int:
        """Max upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    """Build Settings from the current environment."""
    data = Path(_env("DOCMIND_DATA_DIR", "./data")).resolve()
    return Settings(
        data_dir=data,
        upload_dir=data / "uploads",
        qdrant_path=data / "qdrant",
        registry_db=data / "registry.sqlite",
        logs_db=data / "logs.sqlite",
        sheets_db=data / "sheets.sqlite",
        collection=_env("QDRANT_COLLECTION", "docmind_chunks"),
        default_workspace=_env("DEFAULT_WORKSPACE", "default"),
        store_backend=_env("STORE_BACKEND", "qdrant"),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        llm_model=_env("LLM_MODEL", "qwen2.5:3b-instruct"),  # fallback: qwen2.5:1.5b-instruct
        num_ctx=int(_env("LLM_NUM_CTX", "4096")),
        num_predict=int(_env("LLM_NUM_PREDICT", "512")),
        temperature=float(_env("LLM_TEMPERATURE", "0.2")),
        keep_alive=_env("LLM_KEEP_ALIVE", "10m"),
        llm_timeout_s=float(_env("LLM_TIMEOUT_S", "300")),
        embed_model=_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        embed_dim=int(_env("EMBED_DIM", "384")),
        sparse_model=_env("SPARSE_MODEL", "Qdrant/bm25"),
        rerank_model=_env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        embed_batch=int(_env("EMBED_BATCH", "16")),
        fake_models=_bool("DOCMIND_FAKE_MODELS", False),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "25")),
        max_pdf_pages=int(_env("MAX_PDF_PAGES", "500")),
        candidates=int(_env("RETRIEVE_CANDIDATES", "20")),
        top_k=int(_env("RETRIEVE_TOP_K", "4")),
        child_chars=int(_env("CHILD_CHARS", "800")),
        parent_chars=int(_env("PARENT_CHARS", "2000")),
        max_context_tokens=int(_env("MAX_CONTEXT_TOKENS", "3000")),
        # cross-encoder logits: roughly > 0 relevant, < -5 unrelated. Tune per corpus
        # (eval/run_eval.py prints the score distribution to help).
        rerank_threshold=float(_env("RERANK_THRESHOLD", "-3.0")),
        history_turns=int(_env("HISTORY_TURNS", "3")),
        summary_max_sections=int(_env("SUMMARY_MAX_SECTIONS", "8")),
        extract_metadata=_bool("EXTRACT_METADATA", True),
        classifier_min_score=float(_env("CLASSIFIER_MIN_SCORE", "6")),
        classifier_margin=float(_env("CLASSIFIER_MARGIN", "1.5")),
        cache_size=int(_env("CACHE_SIZE", "200")),
        cache_threshold=float(_env("CACHE_THRESHOLD", "0.95")),
        sql_row_limit=int(_env("SQL_ROW_LIMIT", "200")),
        sql_timeout_s=float(_env("SQL_TIMEOUT_S", "5")),
    )


settings: Settings = load_settings()


def reload() -> Settings:
    """Re-read the environment (used by tests and eval)."""
    global settings
    settings = load_settings()
    return settings


def ensure_dirs() -> None:
    """Create data directories if missing."""
    for p in (settings.data_dir, settings.upload_dir, settings.qdrant_path):
        p.mkdir(parents=True, exist_ok=True)
