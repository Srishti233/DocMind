"""Classifier rules (and the LLM only on low confidence)."""
from pathlib import Path

from app.ingest import classifier
from app.ingest.parsers import read_sample
from app.llm import client as llm

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def _cls(name):
    return classifier.classify(read_sample(SAMPLES / name), name)


def test_samples_classified_by_rules_without_llm(stub):
    expect = {"resume_priya_sharma.txt": "resume", "hr_leave_policy.md": "policy",
              "service_agreement.txt": "contract", "api_guide.md": "technical"}
    for name, typ in expect.items():
        c = _cls(name)
        assert (c.doc_type, c.method) == (typ, "rules"), name
    assert stub.n == 0  # rules-first: zero LLM calls


def test_spreadsheet_by_extension(stub):
    assert classifier.classify("", "x.csv").doc_type == "spreadsheet"
    assert classifier.classify("", "x.XLSX").doc_type == "spreadsheet"
    assert stub.n == 0


def test_research_paper_rules(stub):
    text = ("Abstract\nWe propose a novel method. Our approach beats baselines on three datasets "
            "(Smith et al., 2020). Related work. Experiments. References [1] [2]. arXiv:2101.00001")
    assert classifier.classify(text, "paper.txt").doc_type == "research_paper"


def test_low_confidence_calls_llm(stub):
    c = classifier.classify("Lorem ipsum dolor sit amet, nothing recognisable here.", "notes.txt")
    assert stub.n == 1 and c.method in {"llm", "rules-fallback"}
    assert c.doc_type == "other"  # stub answers {"type": "other"}


def test_llm_invalid_output_falls_back(stub):
    stub.handler = lambda m, j: "not json at all"
    c = classifier.classify("Lorem ipsum dolor sit amet.", "notes.txt")
    assert c.doc_type == "other" and c.method == "rules-fallback"
    stub.handler = lambda m, j: '{"type": "banana"}'
    assert classifier.classify("Lorem ipsum dolor.", "n.txt").method == "rules-fallback"


def test_llm_unavailable_falls_back(stub):
    def boom(m, j):
        raise llm.OllamaUnavailable("down")
    stub.handler = boom
    c = classifier.classify("Lorem ipsum dolor sit amet.", "n.txt")
    assert c.doc_type == "other" and c.method == "rules-fallback"


def test_empty_text_skips_llm(stub):
    assert classifier.classify("   ", "n.txt").doc_type == "other"
    assert stub.n == 0
