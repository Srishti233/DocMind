"""The single LLM interface (local Ollama). All LLM calls in DocMind go through here.

Only ONE LLM request runs at a time process-wide: a lock is held for the whole
duration of a call (including streaming). Ingestion runs in a worker thread and
the API in an event loop, so the primitive is a threading lock; the API layer
additionally wraps requests in an asyncio.Semaphore(1) so waiting requests do
not occupy worker threads.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Iterator, Protocol

from app import config


class LLMError(RuntimeError):
    """Base class for LLM failures (message is safe to show to users)."""


class OllamaUnavailable(LLMError):
    """Ollama is not reachable."""


class ModelNotPulled(LLMError):
    """The configured model has not been pulled."""


class Backend(Protocol):
    """Anything that can stream chat completions."""

    def chat(self, messages: list[dict[str, str]], options: dict[str, Any],
             json_mode: bool) -> Iterator[str]:
        """Yield text pieces."""


class OllamaBackend:
    """Streams from Ollama's /api/chat."""

    def chat(self, messages: list[dict[str, str]], options: dict[str, Any],
             json_mode: bool) -> Iterator[str]:
        import httpx  # lazy: only needed when talking to Ollama

        s = config.settings
        payload: dict[str, Any] = {
            "model": s.llm_model, "messages": messages, "stream": True,
            "keep_alive": s.keep_alive, "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            with httpx.stream("POST", f"{s.ollama_url}/api/chat", json=payload,
                              timeout=httpx.Timeout(s.llm_timeout_s, connect=5.0)) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")
                    if r.status_code == 404 or "not found" in body.lower():
                        raise ModelNotPulled(
                            f"Model '{s.llm_model}' is not available in Ollama. "
                            f"Run: ollama pull {s.llm_model}")
                    raise LLMError(f"Ollama returned HTTP {r.status_code}: {body[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("error"):
                        err = str(obj["error"])
                        if "not found" in err.lower():
                            raise ModelNotPulled(
                                f"Model '{s.llm_model}' is not available in Ollama. "
                                f"Run: ollama pull {s.llm_model}")
                        raise LLMError(f"Ollama error: {err}")
                    piece = (obj.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if obj.get("done"):
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaUnavailable(
                f"Ollama is not running at {s.ollama_url}. Start it with `ollama serve`.") from e
        except httpx.ReadTimeout as e:
            raise LLMError("The local model timed out. Try the smaller "
                           "qwen2.5:1.5b-instruct model.") from e


_backend: Backend | None = None
_slot = threading.Lock()  # one LLM request at a time


def set_backend(backend: Backend | None) -> None:
    """Install a backend (tests); None restores Ollama."""
    global _backend
    _backend = backend


def _get_backend() -> Backend:
    return _backend if _backend is not None else OllamaBackend()


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token for English)."""
    return len(text) // 4 + 1


def stream(prompt: str, system: str | None = None, *, json_mode: bool = False,
           num_predict: int | None = None, temperature: float | None = None) -> Iterator[str]:
    """Stream a completion. Holds the global LLM slot until exhausted or closed."""
    s = config.settings
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options = {
        "num_ctx": s.num_ctx,
        "num_predict": num_predict if num_predict is not None else s.num_predict,
        "temperature": temperature if temperature is not None else s.temperature,
    }
    with _slot:
        yield from _get_backend().chat(messages, options, json_mode)


def generate(prompt: str, system: str | None = None, **kw: Any) -> str:
    """Non-streaming completion."""
    return "".join(stream(prompt, system, **kw)).strip()


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of model output, tolerating fences and chatter."""
    text = text.strip()
    for cand in (text, *re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def generate_json(prompt: str, system: str | None = None, num_predict: int = 300) -> dict[str, Any] | None:
    """Ask for a JSON object; returns None if the output cannot be parsed."""
    return extract_json(generate(prompt, system, json_mode=True, num_predict=num_predict,
                                 temperature=0.0))


def check_ollama() -> dict[str, Any]:
    """Report whether Ollama is reachable and the configured model is pulled."""
    import httpx

    s = config.settings
    info: dict[str, Any] = {"reachable": False, "model": s.llm_model, "model_available": False}
    try:
        r = httpx.get(f"{s.ollama_url}/api/tags", timeout=2.0)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        want = s.llm_model if ":" in s.llm_model else s.llm_model + ":latest"
        info["reachable"] = True
        info["model_available"] = want in names
    except Exception:  # noqa: BLE001 - health check must never raise
        pass
    return info
