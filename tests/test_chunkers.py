"""Every chunker + the plugin registry."""
from app.ingest import chunkers
from app.ingest.chunkers import CHUNKERS, Chunk, chunk_document, get_chunker, register
from app.ingest.parsers import Page


def P(text: str) -> list[Page]:
    return [Page(None, text)]


def test_registry_contains_all_types():
    for t in ("resume", "policy", "contract", "technical", "research_paper", "other"):
        assert t in CHUNKERS


def test_adding_a_type_is_one_function():
    @register("recipe")
    def chunk_recipe(pages):
        for p in pages:
            yield Chunk(p.text.upper(), p.number, "Recipe")

    try:
        assert get_chunker("recipe") is chunk_recipe
        assert [c.text for c in chunk_document("recipe", P("soup"))] == ["SOUP"]
    finally:
        del CHUNKERS["recipe"]
    assert get_chunker("recipe") is CHUNKERS["other"]  # unknown types fall back


def test_resume_one_chunk_per_job_and_project():
    text = """# Jane Roe
jane@x.com

EXPERIENCE
Senior Engineer, Acme Corp
2019 - 2022
- Built things
- Led a team

Engineer, Beta Inc
2016 - 2019
- Wrote code

SKILLS
Python, Go, SQL

PROJECTS
Alpha
- Did alpha
Beta
- Did beta
"""
    chunks = list(chunk_document("resume", P(text)))
    labels = [c.section for c in chunks]
    jobs = [c for c in chunks if c.section.startswith("Experience")]
    assert len(jobs) == 2
    assert "Acme Corp" in jobs[0].text and "Beta Inc" not in jobs[0].text
    assert "Beta Inc" in jobs[1].text
    assert sum(l.startswith("Projects") for l in labels) == 2
    assert "Skills" in labels and "Header" in labels


def test_policy_parent_child():
    long_body = " ".join(f"Sentence number {i} about leave rules." for i in range(120))
    text = f"# Policy\n\n## 1. Leave\n{long_body}\n\n## 2. Sick\nShort section about sick days.\n"
    chunks = list(chunk_document("policy", P(text)))
    leave = [c for c in chunks if "Leave" in c.section]
    assert len(leave) > 1  # long section split into several children
    assert len({c.parent_id for c in leave}) >= 1
    for c in leave:
        assert len(c.text) <= 800 + 5
        assert c.parent_text and len(c.parent_text) <= 2000
        assert c.text[:40] in c.parent_text or c.text[:40] in c.parent_text.replace("\n", " ")
    sick = [c for c in chunks if "Sick" in c.section]
    assert len(sick) == 1 and sick[0].parent_text.startswith("Policy > 2. Sick") or "Sick" in sick[0].parent_text


def test_policy_short_section_parent_is_whole_section():
    text = "## 1. A\nOne.\n\nTwo.\n"
    (c,) = list(chunk_document("policy", P(text)))
    assert "One." in c.parent_text and "Two." in c.parent_text


def test_contract_clause_level_keeps_numbering():
    text = """SERVICE AGREEMENT

Between A and B.

1. Services
The Contractor shall provide services.

2. Payment
2.1 Fees. The Client pays $120 per hour.
2.2 Late Payment. Interest is 1.5% per month.

3. Termination
Either party may terminate on 60 days notice.
"""
    chunks = list(chunk_document("contract", P(text)))
    by = {c.section.split(" – ")[0]: c for c in chunks}
    assert "Preamble" in by
    assert by["Clause 2.1"].text.startswith("2. Payment\n2.1 Fees")  # heading carried, numbering kept
    assert by["Clause 2.2"].text.startswith("2.2 Late Payment")
    assert by["Clause 3"].text.startswith("3. Termination")
    assert "Clause 1" in by


def test_technical_never_splits_code_blocks():
    code = "\n".join(f"    line_{i} = compute({i})" for i in range(150))  # ~4000 chars
    text = f"# Guide\n\n## Usage\nIntro paragraph.\n\n```python\n{code}\n```\n\nAfter paragraph.\n\n## Other\nMore text.\n"
    chunks = list(chunk_document("technical", P(text)))
    for c in chunks:
        assert c.text.count("```") % 2 == 0, "a fence was split across chunks"
    code_chunks = [c for c in chunks if "line_0 = compute(0)" in c.text]
    assert len(code_chunks) == 1 and "line_149 = compute(149)" in code_chunks[0].text
    # '#' inside code must not be treated as a heading
    text2 = "# T\n\n```bash\n# comment not heading\necho hi\n```\n"
    (c,) = list(chunk_document("technical", P(text2)))
    assert "# comment not heading" in c.text


def test_research_paper_sections():
    text = """Deep Widgets for Everything
A. Author, B. Author

Abstract— We propose a new widget method that improves accuracy.

1 Introduction
Widgets are important. Prior work [1] studied them.

2 Methods
We train on dataset X.

5 Conclusion
Widgets work.
"""
    chunks = list(chunk_document("research_paper", P(text)))
    sections = [c.section for c in chunks]
    assert "Title" in sections
    abstract = [c for c in chunks if c.section == "Abstract"]
    assert len(abstract) == 1 and "new widget method" in abstract[0].text
    for s in ("Introduction", "Methods", "Conclusion"):
        assert s in sections


def test_other_paragraph_windows_with_overlap():
    paras = [f"Paragraph {i}. " + ("word " * 40).strip() for i in range(20)]
    chunks = list(chunk_document("other", P("\n\n".join(paras))))
    assert len(chunks) > 3
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text[-40:].strip().split()[-3:]
        assert all(w in b.text[:200] for w in tail), "no overlap between consecutive windows"
    assert all(len(c.text) < 800 + 130 for c in chunks)


def test_empty_text_yields_nothing():
    for t in ("resume", "policy", "contract", "technical", "research_paper", "other"):
        assert list(chunk_document(t, P("  \n\n "))) == []


def test_long_paragraph_without_breaks_is_split():
    text = "x" * 5000
    chunks = list(chunk_document("other", P(text)))
    assert len(chunks) >= 6 and all(len(c.text) <= 950 for c in chunks)
