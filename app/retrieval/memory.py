"""Conversation memory: per-session history and follow-up rewriting (rules first)."""
from __future__ import annotations

import re
import threading
from collections import OrderedDict, deque

from app import config
from app.llm import client as llm

_MAX_SESSIONS = 50
_lock = threading.Lock()
_sessions: OrderedDict[str, deque[tuple[str, str]]] = OrderedDict()

_FOLLOWUP = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|he|she|his|her|him|the same|former|latter|"
    r"also|too|instead|what about|how about|and)\b", re.I)


def add_turn(session_id: str | None, question: str, answer: str) -> None:
    """Remember one (standalone question, answer) turn for a session."""
    if not session_id:
        return
    with _lock:
        dq = _sessions.setdefault(session_id, deque(maxlen=config.settings.history_turns))
        dq.append((question[:500], answer[:600]))
        _sessions.move_to_end(session_id)
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)


def get_history(session_id: str | None) -> list[tuple[str, str]]:
    """Last N turns for a session (oldest first)."""
    if not session_id:
        return []
    with _lock:
        return list(_sessions.get(session_id, []))


def clear() -> None:
    """Forget every session (tests)."""
    with _lock:
        _sessions.clear()


def looks_like_followup(question: str) -> bool:
    """Cheap rule: short questions or ones with referring words may need rewriting."""
    q = question.strip()
    return len(q.split()) <= 4 or bool(_FOLLOWUP.search(q))


def rewrite_followup(question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up into a standalone question using the last turns.

    The LLM is called only when there is history AND the question looks like a
    follow-up; any failure returns the original question.
    """
    if not history or not looks_like_followup(question):
        return question
    turns = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-config.settings.history_turns:])
    try:
        out = llm.generate(
            f"Conversation so far:\n{turns}\n\nFollow-up question: {question}\n\n"
            "Rewrite the follow-up as ONE standalone question that makes sense without the "
            "conversation. Keep names and details. Output only the question.",
            system="You rewrite follow-up questions into standalone questions.",
            num_predict=80, temperature=0.0)
    except llm.LLMError:
        return question
    out = out.strip().strip('"').splitlines()[0].strip() if out.strip() else ""
    return out if 3 <= len(out) <= 400 else question
