"""Document-type classifier: rule scoring first, local LLM only when unsure."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app import config
from app.llm import client as llm

DOC_TYPES = ["resume", "policy", "contract", "technical", "research_paper", "spreadsheet", "other"]

_RULES: dict[str, list[tuple[str, float]]] = {
    "resume": [
        (r"\b(?:work|professional) experience\b|\bemployment history\b", 3),
        (r"\beducation\b", 1.5), (r"\bskills\b", 2), (r"\bcertifications?\b", 1),
        (r"\bresume\b|\bcurriculum vitae\b|\bcv\b", 3),
        (r"[\w.+-]+@[\w-]+\.[\w.]+", 1.5),
        (r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current)\b", 2.5),
        (r"\bprojects?\b", 1), (r"linkedin\.com|github\.com", 1),
        (r"\bachievements?\b|\bresponsibilities\b", 1), (r"\bobjective\b|\bsummary\b", 0.5),
    ],
    "policy": [
        (r"\bpolic(?:y|ies)\b", 2.5), (r"\bpurpose\b", 1), (r"\bscope\b", 1.5),
        (r"\bemployees?\b", 1.5), (r"\bcompliance\b", 1.5), (r"\bprocedures?\b", 1.5),
        (r"\beligib\w+", 1.5), (r"\bapproval\b", 1), (r"\bleave\b", 1.5),
        (r"\bdisciplinary\b|\bviolations?\b", 1.5), (r"\bmust\b", 1),
        (r"\bhr\b|\bhuman resources\b", 1),
    ],
    "contract": [
        (r"\bwhereas\b", 3), (r"\bhereinafter\b", 3), (r"\bgoverning law\b", 3),
        (r"\bindemnif\w+", 2), (r"\bterminat\w+", 1), (r"\bthe parties\b|\bthe party\b", 1.5),
        (r"\bagreement\b", 1.5), (r"\bliabilit\w+", 1.5), (r"\bwitness whereof\b", 3),
        (r"\beffective date\b", 2), (r"\bbreach\b", 1.5), (r"\bwarrant\w+", 1),
        (r"\bconfidential\w*", 1), (r"\bclient\b|\bcontractor\b|\bsupplier\b", 1),
        (r"\bshall\b", 0.5), (r"\bsignature\b|\bsigned\b", 1),
    ],
    "technical": [
        (r"\bapi\b", 1.5), (r"\bendpoints?\b", 2), (r"\binstall(?:ation)?\b", 1.5),
        (r"\bconfig(?:uration)?\b", 1.5), (r"\bfunctions?\b|\bclass(?:es)?\b|\bmethods?\b", 1),
        (r"\bparameters?\b", 1.5), (r"```", 3), (r"\bhttp\b|\bjson\b|\bcurl\b", 1.5),
        (r"\bpip install\b|\bnpm \w+|\bdocker\b", 2), (r"\bdef \w+\(|\bimport \w+", 2),
        (r"\barchitecture\b|\bdeploy\w*", 1.5), (r"\bauthentication\b|\btoken\b|\btimeouts?\b", 1),
        (r"\bsdk\b|\bcli\b", 1.5), (r"\brate limit\w*", 1.5),
    ],
    "research_paper": [
        (r"^\s*#*\s*abstract\b", 4), (r"\breferences\b", 1.5), (r"\bet al\.?", 2),
        (r"\barxiv\b", 3), (r"\bwe (?:propose|present|show|introduce)\b", 2.5),
        (r"\bour (?:method|approach|model)\b", 2), (r"\bexperiments?\b", 1.5),
        (r"\bbaselines?\b", 2), (r"\bdatasets?\b", 1), (r"\brelated work\b", 2.5),
        (r"\bconclusions?\b", 1), (r"\bkeywords?\b", 1), (r"\[\d+\]", 1.5), (r"\bdoi\b", 1.5),
    ],
}

_FILENAME_HINTS: dict[str, str] = {
    "resume": r"resume|curriculum|\bcv\b|_cv",
    "policy": r"policy|handbook|procedure|guideline",
    "contract": r"contract|agreement|\bnda\b|\bmsa\b|\bsow\b",
    "technical": r"readme|\bapi\b|manual|spec|docs?\b|guide",
    "research_paper": r"paper|arxiv|study",
}


@dataclass
class Classification:
    """Result of classifying one document."""

    doc_type: str
    method: str  # "extension" | "rules" | "llm" | "rules-fallback"
    scores: dict[str, float] = field(default_factory=dict)


def rule_scores(text: str, filename: str = "") -> dict[str, float]:
    """Score every text type from keyword/regex evidence (counts capped at 3)."""
    scores: dict[str, float] = {}
    for dtype, rules in _RULES.items():
        total = 0.0
        for pattern, weight in rules:
            n = len(re.findall(pattern, text, flags=re.I | re.M))
            total += weight * min(n, 3)
        scores[dtype] = total
    name = filename.lower()
    for dtype, pat in _FILENAME_HINTS.items():
        if re.search(pat, name):
            scores[dtype] += 4
    return scores


def needs_llm(scores: dict[str, float]) -> bool:
    """True when the rules are not decisive (low top score or small margin)."""
    s = config.settings
    ranked = sorted(scores.values(), reverse=True)
    best, second = ranked[0], ranked[1] if len(ranked) > 1 else 0.0
    return best < s.classifier_min_score or (best - second) < s.classifier_margin


def classify(text: str, filename: str = "") -> Classification:
    """Classify a document. Spreadsheets by extension; text by rules, LLM if unsure."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in {"csv", "xlsx"}:
        return Classification("spreadsheet", "extension")
    scores = rule_scores(text, filename)
    best_type = max(scores, key=lambda k: scores[k])
    if not text.strip():
        return Classification("other", "rules", scores)
    if not needs_llm(scores):
        return Classification(best_type, "rules", scores)
    fallback = best_type if scores[best_type] > 0 else "other"
    try:
        out = llm.generate_json(
            "Classify this document into exactly one type from: "
            + ", ".join(t for t in DOC_TYPES if t != "spreadsheet")
            + '.\nReply as JSON: {"type": "<one of the types>"}.\n\n'
            f"Filename: {filename}\n---\n{text[:1500]}\n---",
            system="You classify documents. Output only JSON.", num_predict=40)
    except llm.LLMError:
        return Classification(fallback, "rules-fallback", scores)
    t = str((out or {}).get("type", "")).strip().lower()
    if t in DOC_TYPES and t != "spreadsheet":
        return Classification(t, "llm", scores)
    return Classification(fallback, "rules-fallback", scores)
