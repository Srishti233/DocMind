"""Query router: rules first, the local LLM only when the rules are ambiguous."""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm import client as llm

ROUTES = ("factual", "comparison", "aggregation", "summary", "out_of_scope")

_GREETING = re.compile(r"^\s*(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|bye|ok|okay)\b[\s!.?]*$", re.I)
_OOS = re.compile(
    r"\b(?:who are you|what can you do|what is your name|tell me a joke|weather (?:in|today|like)|"
    r"stock price|news today|write (?:me )?(?:a |an )?(?:poem|song|story|essay|code|script)|"
    r"capital of|who (?:is|was) the (?:president|prime minister)|translate\b)", re.I)
_SUMMARY = re.compile(
    r"\bsummari[sz]e\b|\b(?:summary|overview|gist|main points|key points|key takeaways)\s+of\b|\btl;?dr\b|"
    r"\bwhat(?:'s| is) (?:this|the) (?:document|file|paper|report|policy|contract|resume) about\b", re.I)
_COMPARE = re.compile(r"\b(?:compare|comparison|versus|vs\.?|differences? between|differ(?:s|ence)?|contrast|better than)\b", re.I)
_AGG = re.compile(
    r"\b(?:average|avg|mean|median|sum|total|count|how many|number of|max(?:imum)?|min(?:imum)?|"
    r"highest|lowest|largest|smallest|top \d+|bottom \d+|group by|breakdown)\b", re.I)
_GENERIC = {"name", "date", "id", "value", "type", "total", "number", "count", "the", "of", "and",
            "per", "by", "in", "for", "what", "is", "are", "how", "many", "much"}


@dataclass
class Route:
    """A routing decision."""

    name: str
    reason: str
    source: str  # "rules" | "llm"


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def route_question(question: str, *, has_tables: bool, table_terms: set[str] | None = None,
                   has_text_docs: bool = True) -> Route:
    """Pick a route. ``table_terms`` = table/column/file words of the workspace's spreadsheets."""
    q = question.strip()
    if _GREETING.match(q) or _OOS.search(q):
        return Route("out_of_scope", "chit-chat or unrelated to documents", "rules")
    if _SUMMARY.search(q):
        return Route("summary", "whole-document summary wording", "rules")
    agg = bool(_AGG.search(q))
    if has_tables and agg:
        terms = {t for t in (table_terms or set()) if t not in _GENERIC and len(t) > 2}
        if _words(q) & terms:
            return Route("aggregation", "aggregation wording + spreadsheet term mentioned", "rules")
    if _COMPARE.search(q):
        return Route("comparison", "comparison wording", "rules")
    if has_tables and agg:  # ambiguous: could be a number in a document or a spreadsheet aggregate
        if not has_text_docs:
            return Route("aggregation", "only spreadsheets in workspace", "rules")
        try:
            out = llm.generate_json(
                "Decide how to answer this question.\n"
                '"aggregation" = needs computing (sum/average/count/max/min/group) over spreadsheet tables.\n'
                '"factual" = answer is stated in document text.\n'
                f'Question: {q}\nReply as JSON: {{"route": "aggregation" or "factual"}}',
                system="You route questions. Output only JSON.", num_predict=30)
            r = str((out or {}).get("route", "")).lower()
            if r in {"aggregation", "factual"}:
                return Route(r, "LLM resolved ambiguous aggregation wording", "llm")
        except llm.LLMError:
            pass
        return Route("factual", "ambiguous; LLM unavailable, defaulting to factual", "rules")
    return Route("factual", "default", "rules")
