"""Test doubles: a stub LLM backend that records every call."""
from __future__ import annotations

import re
from typing import Any, Callable, Iterator

Handler = Callable[[list[dict[str, str]], bool], str]


class StubLLM:
    """Backend that answers via ``handler(messages, json_mode)`` and logs calls."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.handler = handler or default_handler
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], options: dict[str, Any], json_mode: bool) -> Iterator[str]:
        self.calls.append({"messages": messages, "json": json_mode, "options": options})
        text = self.handler(messages, json_mode)
        for part in re.findall(r"\S+\s*", text):
            yield part

    @property
    def n(self) -> int:
        return len(self.calls)

    def last_prompt(self) -> str:
        return self.calls[-1]["messages"][-1]["content"]


def default_handler(messages: list[dict[str, str]], json_mode: bool) -> str:
    """Plausible canned behaviour for every LLM task in the app."""
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    prompt = messages[-1]["content"]
    if "You classify documents" in system:
        return '{"type": "other"}'
    if "You extract structured data" in system:
        if "resume" in prompt[:80]:
            return '{"name": "Priya Sharma", "skills": ["Python", "SQL"], "years_experience": 8, "companies": ["Northwind Analytics", "Contoso Labs"]}'
        return '{"parties": ["BlueRiver Consulting Ltd", "Helios Retail Inc"], "dates": ["1 March 2025"]}'
    if "You route questions" in system:
        q = prompt.split("Question:", 1)[-1].lower()
        spreadsheet_words = ("sales", "units", "region", "price", "revenue", "spreadsheet")
        return '{"route": "aggregation"}' if any(w in q for w in spreadsheet_words) else '{"route": "factual"}'
    if "rewrite follow-up" in system:
        return "How many days of sick leave do employees receive?"
    if "SQLite SELECT" in system:
        q = prompt.lower()
        if "per region" in q or "by region" in q:
            return "```sql\nSELECT region, SUM(units_sold) AS total_units FROM sales GROUP BY region ORDER BY region;\n```"
        if "average" in q:
            return "SELECT AVG(unit_price) AS avg_price FROM sales"
        return "SELECT SUM(units_sold) AS total_units FROM sales"
    if "<<<SOURCE" in prompt:
        return "Based on the sources, see [1]."
    return "ok"
