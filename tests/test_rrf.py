"""Reciprocal Rank Fusion."""
from app.retrieval.hybrid import rrf_fuse


def test_known_scores():
    out = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert abs(out["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(out["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(out["c"] - 1 / 63) < 1e-12 and abs(out["d"] - 1 / 63) < 1e-12


def test_agreement_beats_single_list_leader():
    fused = [i for i, _ in rrf_fuse([["x", "a", "b"], ["y", "a", "c"]])]
    assert fused[0] == "a"  # ranked 2nd in both lists beats 1st in only one


def test_deterministic_tie_break_by_first_appearance():
    fused = [i for i, _ in rrf_fuse([["a", "b"], ["b", "a"]])]
    assert fused == ["a", "b"]


def test_empty_and_single_lists():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
    assert [i for i, _ in rrf_fuse([["q", "r"]])] == ["q", "r"]


def test_k_changes_scale_not_order():
    a = [i for i, _ in rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=1)]
    b = [i for i, _ in rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=1000)]
    assert a == b
