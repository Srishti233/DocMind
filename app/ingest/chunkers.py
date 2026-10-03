"""Type-specific chunkers behind a plugin registry.

Adding a document type = write one generator function decorated with
``@register("my_type")`` that takes an iterable of ``Page`` and yields ``Chunk``.
Chunkers are generators over a line stream: only the current section is buffered,
never the whole document's chunks.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from app import config
from app.ingest.parsers import Page


@dataclass
class Chunk:
    """One indexable unit of text."""

    text: str
    page: int | None
    section: str
    parent_id: str | None = None
    parent_text: str | None = None


ChunkerFn = Callable[[Iterable[Page]], Iterator[Chunk]]
CHUNKERS: dict[str, ChunkerFn] = {}


def register(doc_type: str) -> Callable[[ChunkerFn], ChunkerFn]:
    """Decorator that registers a chunker for ``doc_type``."""
    def deco(fn: ChunkerFn) -> ChunkerFn:
        CHUNKERS[doc_type] = fn
        return fn
    return deco


def get_chunker(doc_type: str) -> ChunkerFn:
    """Chunker for a type, falling back to the generic paragraph chunker."""
    return CHUNKERS.get(doc_type, CHUNKERS["other"])


def chunk_document(doc_type: str, pages: Iterable[Page]) -> Iterator[Chunk]:
    """Chunk a page stream with the right chunker, dropping empty chunks."""
    for c in get_chunker(doc_type)(pages):
        if c.text and c.text.strip():
            yield c


# --------------------------------------------------------------------------- helpers

def iter_lines(pages: Iterable[Page]) -> Iterator[tuple[int | None, str]]:
    """Flatten pages into (page_number, line)."""
    for p in pages:
        for line in p.text.split("\n"):
            yield p.number, line.rstrip()


def _split_para(p: str, limit: int) -> list[str]:
    """Split one oversized paragraph on lines, then sentences, then hard cuts."""
    if len(p) <= limit:
        return [p]
    if "\n" in p:
        return _pack(p.split("\n"), limit, "\n")
    out: list[str] = []
    cur = ""
    for s in re.split(r"(?<=[.!?])\s+", p):
        if len(s) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(s[i:i + limit] for i in range(0, len(s), limit))
        elif cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _pack(blocks: list[str], limit: int, sep: str) -> list[str]:
    out: list[str] = []
    cur = ""
    for b in blocks:
        if not b.strip():
            continue
        if len(b) > limit:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_split_para(b, limit))
        elif cur and len(cur) + len(sep) + len(b) > limit:
            out.append(cur)
            cur = b
        else:
            cur = f"{cur}{sep}{b}" if cur else b
    if cur:
        out.append(cur)
    return out


def split_long(text: str, limit: int) -> list[str]:
    """Split text on paragraph boundaries into pieces of at most ``limit`` chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _pack(paras, limit, "\n\n")


_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUM = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+([A-Z][^\n]{1,80})$")
_KW = re.compile(r"^(?:section|article|chapter|part)\s+(?:\d+|[IVX]+)\b", re.I)


def heading_level(line: str, allow_caps: bool = True) -> tuple[int, str] | None:
    """Detect a heading line. Returns (level, title) or None."""
    s = line.strip()
    if not s or len(s) > 100:
        return None
    m = _MD.match(s)
    if m:
        return len(m.group(1)), m.group(2)
    if s.endswith((".", ",", ";")):
        return None
    m = _NUM.match(s)
    if m:
        return m.group(1).count(".") + 1, s
    if _KW.match(s):
        return 1, s
    if (allow_caps and s.isupper() and 3 <= len(s) <= 70 and re.search(r"[A-Z]{3}", s)
            and not re.fullmatch(r"[\d\W]+", s)):
        return 1, s.title()
    return None


def _sections(pages: Iterable[Page], heading_fn: Callable[[str], tuple[int, str] | None],
              default_title: str, track_code: bool = False
              ) -> Iterator[tuple[list[str], int | None, list[str]]]:
    """Group a line stream into (heading_path, page, body_lines) sections."""
    stack: list[tuple[int, str]] = []
    path = [default_title]
    lines: list[str] = []
    sec_page: int | None = None
    in_code = False
    for page, line in iter_lines(pages):
        is_fence = track_code and line.strip().startswith("```")
        if is_fence:
            in_code = not in_code
        h = None if (in_code or is_fence) else heading_fn(line)
        if h:
            if any(x.strip() for x in lines):
                yield path, sec_page, lines
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path, lines, sec_page = [t for _, t in stack], [], page
        else:
            if sec_page is None and line.strip():
                sec_page = page
            lines.append(line)
    if any(x.strip() for x in lines):
        yield path, sec_page, lines


# --------------------------------------------------------------------------- chunkers

@register("other")
def chunk_other(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Paragraph windows with overlap (generic fallback)."""
    size = config.settings.child_chars
    overlap = min(120, size // 5)

    def paragraphs() -> Iterator[tuple[int | None, str]]:
        buf: list[str] = []
        first: int | None = None
        for page, line in iter_lines(pages):
            if line.strip():
                if not buf:
                    first = page
                buf.append(line)
            elif buf:
                yield first, "\n".join(buf)
                buf = []
        if buf:
            yield first, "\n".join(buf)

    window = ""
    wpage: int | None = None
    for page, para in paragraphs():
        for piece in _split_para(para, size):
            if window and len(window) + 2 + len(piece) > size:
                yield Chunk(window, wpage, "Body")
                tail = window[-overlap:]
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                window, wpage = tail, page
            if not window:
                wpage = page
            window = f"{window}\n\n{piece}" if window else piece
    if window.strip():
        yield Chunk(window, wpage, "Body")


_RESUME_SECTIONS = {
    "summary", "professional summary", "profile", "objective", "experience", "work experience",
    "professional experience", "employment history", "education", "skills", "technical skills",
    "projects", "certifications", "awards", "publications", "languages", "interests",
    "achievements",
}
_RESUME_ENTRY_SECTIONS = {"experience", "work experience", "professional experience",
                          "employment history", "projects"}
_BULLET = re.compile(r"^[-•*·▪●\u2022]\s+")


@register("resume")
def chunk_resume(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Section-based; one chunk per job / project inside experience-like sections."""
    limit = config.settings.child_chars * 2
    title, entry_mode = "Header", False
    cur: list[str] = []
    cur_page: int | None = None
    has_bullets = False
    blank_before = False

    def flush() -> list[Chunk]:
        text = "\n".join(cur).strip()
        out: list[Chunk] = []
        if text:
            label = f"{title} – {cur[0][:80]}" if entry_mode and cur else title
            out = [Chunk(p, cur_page, label) for p in split_long(text, limit)]
        return out

    for page, raw in iter_lines(pages):
        line = raw.strip()
        name = line.lstrip("#").strip().rstrip(":").lower() if len(line) < 40 else ""
        if name in _RESUME_SECTIONS:
            yield from flush()
            cur, has_bullets = [], False
            title, entry_mode = line.lstrip("#").strip().rstrip(":").title(), name in _RESUME_ENTRY_SECTIONS
            blank_before = False
            continue
        if not line:
            blank_before = True
            if not entry_mode and cur:
                cur.append("")
            continue
        is_bullet = bool(_BULLET.match(line))
        if entry_mode and not is_bullet and cur and (has_bullets or (blank_before and len(cur) >= 2)):
            yield from flush()
            cur, has_bullets = [], False
        if not cur:
            cur_page = page
        cur.append(line)
        has_bullets = has_bullets or is_bullet
        blank_before = False
    yield from flush()


def _parent_for(body: str, piece: str, label: str, base_id: str, parent_chars: int) -> tuple[str, str]:
    """Parent text for a child: the whole section, or a window around the child."""
    full = f"{label}\n{body}"
    if len(full) <= parent_chars:
        return base_id, full
    idx = max(0, full.find(piece[:60]))
    start = max(0, min(idx - (parent_chars - len(piece)) // 2, len(full) - parent_chars))
    return f"{base_id}-{start // max(1, parent_chars // 2)}", full[start:start + parent_chars]


@register("policy")
def chunk_policy(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Heading hierarchy; search small child chunks, hand the parent section to the LLM."""
    s = config.settings
    for path, page, lines in _sections(pages, heading_level, "Overview"):
        body = "\n".join(lines).strip()
        if not body:
            continue
        label = " > ".join(path)
        base = uuid.uuid4().hex[:12]
        for piece in split_long(body, s.child_chars):
            pid, ptext = _parent_for(body, piece, label, base, s.parent_chars)
            yield Chunk(piece, page, label, pid, ptext)


_CLAUSE = re.compile(
    r"^\s*(?:(?:article|section|clause)\s+(\d+(?:\.\d+)*)[.:)]?|(\d+\.\d+(?:\.\d+)*)\.?|(\d+)[.)])\s+(\S.*)$",
    re.I)


@register("contract")
def chunk_contract(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Clause-level chunks that keep the original numbering in text and label."""
    limit = config.settings.child_chars * 2
    num: str | None = None
    label = "Preamble"
    cur: list[str] = []
    cur_page: int | None = None
    carry = ""  # a bare heading like "3. PAYMENT" is prepended to its first sub-clause

    def flush() -> list[Chunk]:
        nonlocal carry
        text = "\n".join(cur).strip()
        if not text:
            return []
        if num and len(cur) == 1 and len(text) < 60 and not text.rstrip().endswith((".", ";", ":", ",")):
            carry = text  # bare heading such as "2. Payment"; merged into the next clause
            return []
        if carry:
            text, carry = f"{carry}\n{text}", ""
        parts = split_long(text, limit) if "\n\n" in text else _pack(text.split("\n"), limit, "\n")
        return [Chunk(p, cur_page, label if i == 0 else f"{label} (cont.)")
                for i, p in enumerate(parts)]

    for page, line in iter_lines(pages):
        m = _CLAUSE.match(line)
        if m:
            yield from flush()
            num = m.group(1) or m.group(2) or m.group(3)
            head = re.split(r"[.:]", m.group(4), maxsplit=1)[0].strip()
            label = f"Clause {num}" + (f" – {head}" if 0 < len(head) <= 40 else "")
            cur, cur_page = [line.strip()], page
        else:
            if not cur and line.strip():
                cur_page = page
            if line.strip() or cur:
                cur.append(line.strip())
    yield from flush()
    if carry:  # a trailing bare heading with nothing after it
        yield Chunk(carry, cur_page, label)


def _blocks_with_code(lines: list[str]) -> list[tuple[str, bool]]:
    """Split lines into (text, is_code) blocks; fenced code stays atomic."""
    blocks: list[tuple[str, bool]] = []
    cur: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            if not in_code:
                if cur:
                    blocks.append(("\n".join(cur).strip(), False))
                cur, in_code = [line], True
            else:
                cur.append(line)
                blocks.append(("\n".join(cur), True))
                cur, in_code = [], False
            continue
        if in_code:
            cur.append(line)
        elif not line.strip():
            if cur:
                blocks.append(("\n".join(cur).strip(), False))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(("\n".join(cur).strip() if not in_code else "\n".join(cur), in_code))
    return [b for b in blocks if b[0].strip()]


@register("technical")
def chunk_technical(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Heading-based; fenced code blocks are never split."""
    limit = int(config.settings.child_chars * 1.5)
    nocaps = lambda l: heading_level(l, allow_caps=False)  # noqa: E731
    for path, page, lines in _sections(pages, nocaps, "Overview", track_code=True):
        label = " > ".join(path)
        cur = ""
        for text, is_code in _blocks_with_code(lines):
            if is_code and len(text) > limit:  # oversized code block: keep whole, alone
                if cur:
                    yield Chunk(cur, page, label)
                    cur = ""
                yield Chunk(text, page, label)
                continue
            pieces = [text] if is_code else _split_para(text, limit)
            for piece in pieces:
                if cur and len(cur) + 2 + len(piece) > limit:
                    yield Chunk(cur, page, label)
                    cur = piece
                else:
                    cur = f"{cur}\n\n{piece}" if cur else piece
        if cur.strip():
            yield Chunk(cur, page, label)


_PAPER_NAMES = (r"abstract|introduction|related work|background|methods?|methodology|approach|"
                r"experiments?|experimental setup|results?|evaluation|discussion|limitations|"
                r"conclusions?|references|acknowledg(?:e)?ments?")
_PAPER_HEAD = re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?({_PAPER_NAMES})\s*:?$", re.I)


def _paper_heading(line: str) -> tuple[int, str] | None:
    s = line.strip()
    m = _PAPER_HEAD.match(s.lstrip("#").strip())
    if m:
        return 1, m.group(1).title()
    h = heading_level(line, allow_caps=False)
    return (2, h[1]) if h else None


@register("research_paper")
def chunk_paper(pages: Iterable[Page]) -> Iterator[Chunk]:
    """Abstract as its own chunk, then section-by-section windows."""
    limit = int(config.settings.child_chars * 1.5)

    def prepped() -> Iterator[Page]:  # "Abstract— text" -> "Abstract\ntext"
        for p in pages:
            yield Page(p.number, re.sub(r"(?im)^(abstract)\s*[:—–-]\s+", r"\1\n", p.text))

    for path, page, lines in _sections(prepped(), _paper_heading, "Title"):
        body = "\n".join(lines).strip()
        label = " > ".join(path)
        for i, piece in enumerate(split_long(body, limit)):
            yield Chunk(piece, page, label if i == 0 else f"{label} (cont.)")
