"""Cache, memory, router, parsers, LLM helpers, config."""
import pytest

from app import config
from app.ingest import parsers
from app.llm import client as llm
from app.retrieval import memory, router
from app.retrieval.cache import CacheEntry, SemanticCache
from app.retrieval.summarize import plan_groups
from app.retrieval.store import Hit


def _entry(key, vec, ans="a"):
    return CacheEntry(key, vec, ans, [], "factual")


def test_cache_cosine_threshold_and_key_isolation():
    c = SemanticCache()
    k = ("ws", "", "")
    c.put(_entry(k, [1.0, 0.0]))
    assert c.get(k, [1.0, 0.0]) is not None
    assert c.get(k, [0.9, 0.436]) is None  # cosine 0.9 < 0.95
    assert c.get(("other", "", ""), [1.0, 0.0]) is None
    assert c.get(("ws", "policy", ""), [1.0, 0.0]) is None


def test_cache_capped_at_200_with_lru_eviction():
    c = SemanticCache()
    k = ("ws", "", "")
    assert config.settings.cache_size == 200
    for i in range(200):
        v = [0.0] * 201
        v[i] = 1.0
        c.put(_entry(k, v, str(i)))
    v0 = [1.0] + [0.0] * 200
    assert c.get(k, v0).answer == "0"  # touch entry 0 -> most recently used
    extra = [0.0] * 200 + [1.0]
    c.put(_entry(k, extra, "new"))
    assert len(c) == 200
    assert c.get(k, v0) is not None  # survived
    v1 = [0.0, 1.0] + [0.0] * 199
    assert c.get(k, v1) is None  # entry 1 was the LRU victim
    c.invalidate("ws")
    assert len(c) == 0


def test_followup_rules_skip_llm_when_not_needed(stub):
    hist = [("What is annual leave?", "24 days [1].")]
    q = "How many days of sick leave do employees receive per year?"
    assert memory.rewrite_followup(q, hist) == q and stub.n == 0
    assert memory.rewrite_followup("what about sick leave?", []) == "what about sick leave?" and stub.n == 0
    assert memory.rewrite_followup("what about sick leave?", hist) != "what about sick leave?"
    assert stub.n == 1


def test_memory_keeps_last_three_turns():
    for i in range(5):
        memory.add_turn("s", f"q{i}", f"a{i}")
    assert [q for q, _ in memory.get_history("s")] == ["q2", "q3", "q4"]
    assert memory.get_history(None) == [] and memory.get_history("zzz") == []


def test_rewrite_failure_returns_original(stub):
    def boom(m, j):
        raise llm.OllamaUnavailable("x")
    stub.handler = boom
    assert memory.rewrite_followup("and him?", [("q", "a")]) == "and him?"


def test_router_rules():
    r = router.route_question
    assert r("hello", has_tables=False).name == "out_of_scope"
    assert r("Summarize the contract", has_tables=False).name == "summary"
    assert r("Compare the two policies", has_tables=False).name == "comparison"
    assert r("What is the average price?", has_tables=True, table_terms={"sales", "price"}).name == "aggregation"
    assert r("What is the notice period?", has_tables=True, table_terms={"sales"}).name == "factual"
    assert r("average anything", has_tables=True, table_terms={"sales"}, has_text_docs=False).name == "aggregation"
    assert r("total units", has_tables=False).name == "factual"  # no tables -> never SQL


def test_router_ambiguous_llm_unavailable_defaults_to_factual(stub):
    def boom(m, j):
        raise llm.OllamaUnavailable("x")
    stub.handler = boom
    d = router.route_question("how many people attended?", has_tables=True, table_terms={"sales"})
    assert d.name == "factual"


def test_summary_groups_capped_at_8():
    chunks = [Hit(str(i), 0, {"text": "x" * 800, "page": i, "section": "S", "seq": i, "filename": "f", "doc_id": "d"})
              for i in range(200)]
    groups = plan_groups(chunks)
    assert 1 < len(groups) <= 8 and all(len(g.text) <= 3200 for g in groups)
    assert plan_groups([]) == []


def test_extract_json_tolerant():
    assert llm.extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert llm.extract_json("nope") is None
    assert llm.extract_json("[1,2]") is None


def test_global_llm_slot_serialises_requests(stub):
    import threading, time
    active, peak = [0], [0]
    lock = threading.Lock()

    def slow(m, j):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return "ok"
    stub.handler = slow
    ts = [threading.Thread(target=llm.generate, args=("hi",)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert peak[0] == 1


def test_defaults_match_8gb_profile():
    s = config.settings
    assert s.llm_model == "qwen2.5:3b-instruct" and s.num_ctx == 4096 and s.num_predict == 512
    assert (s.candidates, s.top_k, s.embed_batch, s.cache_size) == (20, 4, 16, 200)
    assert (s.child_chars, s.parent_chars, s.max_upload_mb, s.max_pdf_pages) == (800, 2000, 25, 500)
    assert s.keep_alive == "10m" and s.temperature == 0.2 and s.extract_metadata is True


def test_parse_unsupported_and_text(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n\n\n\nworld")
    pages = list(parsers.parse_pages(tmp_path / "a.txt"))
    assert pages[0].text == "hello\n\nworld"
    with pytest.raises(parsers.ParseError):
        parsers.parse_pages(tmp_path / "a.exe")


def test_dotenv_loader(tmp_path, monkeypatch):
    """.env values load, quotes/comments are handled, real environment variables win."""
    from app import config

    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_A=plain\nFOO_B=\"quoted value\"\nexport FOO_C=exported # note\n"
                   "FOO_D=from_file\n\nnot a pair\n", encoding="utf-8")
    for k in ("FOO_A", "FOO_B", "FOO_C"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FOO_D", "from_env")
    config._load_dotenv(env)
    import os
    assert (os.environ["FOO_A"], os.environ["FOO_B"], os.environ["FOO_C"]) == ("plain", "quoted value", "exported")
    assert os.environ["FOO_D"] == "from_env"
    for k in ("FOO_A", "FOO_B", "FOO_C"):
        monkeypatch.delenv(k, raising=False)
    config._load_dotenv(tmp_path / "missing.env")  # no file: no error
