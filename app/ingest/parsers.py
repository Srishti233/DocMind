"""File parsers. Each yields ``Page`` objects lazily so large files stream."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from app import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"}
SPREADSHEET_EXTENSIONS = {".csv", ".xlsx"}


class ParseError(Exception):
    """A file could not be parsed; the message is safe to show to users."""


@dataclass(frozen=True)
class Page:
    """A unit of extracted text. ``number`` is None for formats without pages."""

    number: int | None
    text: str


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def parse_pdf(path: Path) -> Iterator[Page]:
    """Stream a PDF page by page. Raises ParseError for scanned/encrypted/huge PDFs."""
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ParseError("The PDF is password protected.")
        n = len(reader.pages)
    except ParseError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"Could not read the PDF: {e}") from e
    limit = config.settings.max_pdf_pages
    if n > limit:
        raise ParseError(f"The PDF has {n} pages; the limit is {limit}. "
                         "Split it or raise MAX_PDF_PAGES.")
    any_text = False
    for i in range(n):
        try:
            text = _clean(reader.pages[i].extract_text() or "")
        except Exception:  # noqa: BLE001 - a bad page should not kill the document
            text = ""
        if text.strip():
            any_text = True
        yield Page(i + 1, text)
    if not any_text:
        raise ParseError("No extractable text found. This looks like a scanned PDF; "
                         "OCR is intentionally not supported.")


def parse_docx(path: Path) -> Iterator[Page]:
    """Stream a DOCX in blocks; headings become markdown '#', tables become pipe rows."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(str(path))
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"Could not read the DOCX: {e}") from e
    buf: list[str] = []
    size = 0
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                buf.append("")
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if m:
                text = "#" * int(m.group(1)) + " " + text
            elif style == "Title":
                text = "# " + text
            elif style.lower().startswith("list"):
                text = "- " + text
            buf.append(text)
            size += len(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                line = " | ".join(c.text.strip() for c in row.cells)
                buf.append(line)
                size += len(line)
            buf.append("")
        if size > 8000:
            yield Page(None, _clean("\n".join(buf)))
            buf, size = [], 0
    if buf:
        yield Page(None, _clean("\n".join(buf)))


def parse_text(path: Path) -> Iterator[Page]:
    """Stream TXT/MD in ~20k character blocks split on line boundaries."""
    buf: list[str] = []
    size = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= 20000:
                yield Page(None, _clean("".join(buf)))
                buf, size = [], 0
    if buf:
        yield Page(None, _clean("".join(buf)))


def parse_pages(path: Path) -> Iterator[Page]:
    """Dispatch on extension. Spreadsheets are handled by ingest.spreadsheet instead."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".txt", ".md"}:
        return parse_text(path)
    raise ParseError(f"Unsupported file type '{ext}'. Supported: "
                     + ", ".join(sorted(SUPPORTED_EXTENSIONS)))


def read_sample(path: Path, max_pages: int = 6, max_chars: int = 6000) -> str:
    """Read the leading text of a file (for classification / metadata)."""
    out: list[str] = []
    total = 0
    gen = parse_pages(path)
    try:
        for i, page in enumerate(gen):
            out.append(page.text)
            total += len(page.text)
            if i + 1 >= max_pages or total >= max_chars:
                break
    finally:
        gen.close()
    return "\n".join(out)[:max_chars]
