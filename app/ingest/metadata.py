"""Structured metadata extraction (resume, contract) via local LLM JSON + safe fallback."""
from __future__ import annotations

import datetime
import re
from typing import Any, Callable

from app import config
from app.llm import client as llm

EXTRACTABLE_TYPES = {"resume", "contract"}


def _str_list(v: Any, max_items: int = 30) -> list[str]:
    if isinstance(v, str):
        v = [x for x in re.split(r"[,;\n]", v)]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if isinstance(x, (str, int, float)) and str(x).strip():
            out.append(str(x).strip()[:120])
    return out[:max_items]


def _validate_resume(d: dict[str, Any]) -> dict[str, Any]:
    name = d.get("name")
    years = d.get("years_experience")
    try:
        years = round(float(years), 1) if years is not None and str(years).strip() != "" else None
    except (TypeError, ValueError):
        years = None
    if years is not None and not (0 <= years <= 60):
        years = None
    return {"name": str(name).strip()[:120] if isinstance(name, str) and name.strip() else None,
            "skills": _str_list(d.get("skills")), "years_experience": years,
            "companies": _str_list(d.get("companies"))}


def _validate_contract(d: dict[str, Any]) -> dict[str, Any]:
    return {"parties": _str_list(d.get("parties"), 10), "dates": _str_list(d.get("dates"), 10)}


def _fallback_resume(text: str) -> dict[str, Any]:
    first = next((l.strip("# ").strip() for l in text.splitlines() if l.strip()), "")
    skills: list[str] = []
    m = re.search(r"(?im)^\s*#*\s*(?:technical\s+)?skills\s*:?\s*\n?(.+)$", text)
    if m:
        skills = _str_list(m.group(1))
    spans = re.findall(r"\b((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2}|present|current)\b", text, re.I)
    years = None
    if spans:
        total = 0
        for a, b in spans:
            end = datetime.date.today().year if b.lower() in {"present", "current"} else int(b)
            total += max(0, end - int(a))
        years = float(total) if total <= 60 else None
    return {"name": first[:120] or None, "skills": skills, "years_experience": years,
            "companies": []}


def _fallback_contract(text: str) -> dict[str, Any]:
    parties: list[str] = []
    m = re.search(r"between\s+(.+?)\s+(?:\(.*?\)\s*)?and\s+(.+?)[\s(,.]", text[:2000], re.I | re.S)
    if m:
        parties = [m.group(1).strip()[:100], m.group(2).strip()[:100]]
    dates = re.findall(r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|"
                       r"September|October|November|December)\s+(?:\d{1,2},\s*)?\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
                       text[:4000])
    return {"parties": parties, "dates": list(dict.fromkeys(dates))[:10]}


_SCHEMAS: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]],
                           Callable[[str], dict[str, Any]]]] = {
    "resume": ('{"name": str, "skills": [str], "years_experience": number, "companies": [str]}',
               _validate_resume, _fallback_resume),
    "contract": ('{"parties": [str], "dates": [str]}', _validate_contract, _fallback_contract),
}


def extract_metadata(doc_type: str, text: str) -> dict[str, Any]:
    """Extract metadata for resume/contract documents.

    Never raises: invalid or missing LLM output falls back to rule-based extraction.
    Returns {} for other types or when EXTRACT_METADATA=false.
    """
    if doc_type not in EXTRACTABLE_TYPES or not config.settings.extract_metadata:
        return {}
    schema, validate, fallback = _SCHEMAS[doc_type]
    sample = text[:5000]
    try:
        raw = llm.generate_json(
            f"Extract these fields from the {doc_type} below as JSON with exactly this shape: "
            f"{schema}. Use null or [] when unknown. Do not invent values.\n---\n{sample}\n---",
            system="You extract structured data. Output only JSON.", num_predict=300)
        if raw:
            out = validate(raw)
            out["_source"] = "llm"
            return out
    except llm.LLMError:
        pass
    out = validate(fallback(sample))
    out["_source"] = "fallback"
    return out


def metadata_chunk_text(doc_type: str, filename: str, meta: dict[str, Any]) -> str:
    """Searchable text for the synthetic 'extracted metadata' chunk."""
    if not meta:
        return ""
    if doc_type == "resume":
        return (f"Profile summary of {filename}. Name: {meta.get('name') or 'unknown'}. "
                f"Skills: {', '.join(meta.get('skills', [])) or 'n/a'}. "
                f"Years of experience: {meta.get('years_experience') if meta.get('years_experience') is not None else 'n/a'}. "
                f"Companies: {', '.join(meta.get('companies', [])) or 'n/a'}.")
    if doc_type == "contract":
        return (f"Contract details of {filename}. Parties: {', '.join(meta.get('parties', [])) or 'n/a'}. "
                f"Dates: {', '.join(meta.get('dates', [])) or 'n/a'}.")
    return ""
