"""Map-reduce summarisation (at most SUMMARY_MAX_SECTIONS sections per document)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from app import config
from app.llm import client as llm
from app.retrieval.hybrid import Source
from app.retrieval.injection import SYSTEM_PROMPT, wrap_source
from app.retrieval.store import Hit

GROUP_CHARS = 3200  # ~800 tokens per map call


@dataclass
class Group:
    """A contiguous run of chunks summarised together."""

    text: str
    page_start: int | None
    page_end: int | None
    label: str
    chunk_id: str
    doc_id: str
    filename: str


def plan_groups(chunks: list[Hit], max_sections: int | None = None) -> list[Group]:
    """Partition ordered chunks into at most ``max_sections`` contiguous groups."""
    if not chunks:
        return []
    cap = max_sections or config.settings.summary_max_sections
    total = sum(len(c.payload.get("text", "")) for c in chunks)
    g = max(1, min(cap, math.ceil(total / GROUP_CHARS)))
    target = total / g
    groups: list[list[Hit]] = [[]]
    acc = 0.0
    for c in chunks:
        if acc >= target * len(groups) and len(groups) < g:
            groups.append([])
        groups[-1].append(c)
        acc += len(c.payload.get("text", ""))
    out = []
    for grp in groups:
        if not grp:
            continue
        pages = [c.payload.get("page") for c in grp if c.payload.get("page") is not None]
        sections = list(dict.fromkeys(c.payload.get("section", "") for c in grp if c.payload.get("section")))
        text = "\n\n".join(c.payload.get("text", "") for c in grp)[:GROUP_CHARS]
        out.append(Group(text, min(pages) if pages else None, max(pages) if pages else None,
                         " / ".join(sections[:3]) or "Body", grp[0].id,
                         grp[0].payload.get("doc_id", ""), grp[0].payload.get("filename", "?")))
    return out


def group_sources(groups: list[Group]) -> list[Source]:
    """One numbered Source per group (used for citations)."""
    return [Source(i + 1, g.filename, g.page_start, g.label, g.text[:300], g.text[:2200], g.doc_id,
                   g.chunk_id, 0.0, False, "") for i, g in enumerate(groups)]


def map_group(group: Group, n: int) -> str:
    """Summarise one group in <= 70 words."""
    block = wrap_source(n, group.filename, group.page_start, group.label, group.text, False)
    return llm.generate(
        f"{block}\n\nSummarize SOURCE {n} in at most 70 words. Keep key facts, names and numbers.",
        system=SYSTEM_PROMPT, num_predict=140, temperature=0.1)


def reduce_stream(question: str, filename: str, partials: list[str]) -> Iterator[str]:
    """Stream the final summary built from per-section summaries."""
    blocks = "\n\n".join(wrap_source(i + 1, filename, None, f"section {i + 1} summary", p, False)
                         for i, p in enumerate(partials))
    yield from llm.stream(
        f"{blocks}\n\nTask: {question}\nWrite one coherent summary of the whole document "
        "(5 sentences or fewer) and cite the section numbers [n] you used.",
        system=SYSTEM_PROMPT)
