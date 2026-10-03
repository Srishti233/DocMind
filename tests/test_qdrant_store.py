"""The real embedded Qdrant store (no models or network needed).

The rest of the suite uses the in-memory store; this file guards the production backend.
"""
from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from app.retrieval.store import Point, QdrantStore  # noqa: E402

DIM = 4


def _pt(n: int, ws: str, doc: str, seq: int, sparse: tuple[list[int], list[float]] = ([], []),
        dense=(1.0, 0.0, 0.0, 0.0), **payload) -> Point:
    return Point(f"00000000-0000-0000-0000-{n:012d}", list(dense), sparse[0], sparse[1],
                 {"workspace": ws, "doc_id": doc, "seq": seq, "text": f"chunk {n}", **payload})


@pytest.fixture
def qs(tmp_path):
    return QdrantStore(str(tmp_path / "q"), "t", DIM)


def test_sparse_search_applies_idf_so_rare_terms_win(qs):
    """Regression: without the IDF modifier a very common term outranks a rare one."""
    common, rare = 1, 2
    qs.upsert([
        _pt(1, "w", "a", 0, ([common], [3.0])),
        _pt(2, "w", "b", 0, ([common], [3.0])),
        _pt(3, "w", "c", 0, ([common, rare], [1.0, 1.0])),  # only doc containing the rare term
        _pt(4, "w", "d", 0, ([common], [2.0])),
        _pt(5, "w", "e", 0, ([common], [2.0])),
    ])
    hits = qs.search_sparse([common, rare], [1.0, 1.0], "w", 5)
    assert hits[0].payload["doc_id"] == "c"


def test_workspace_and_metadata_filters(qs):
    qs.upsert([
        _pt(1, "w1", "a", 0, doc_type="policy", filename="a.md"),
        _pt(2, "w1", "b", 0, dense=(0.9, 0.1, 0, 0), doc_type="resume", filename="b.txt"),
        _pt(3, "w2", "c", 0, doc_type="policy", filename="c.md"),
    ])
    vec = [1.0, 0.0, 0.0, 0.0]
    assert {h.payload["doc_id"] for h in qs.search_dense(vec, "w1", 10)} == {"a", "b"}
    assert [h.payload["doc_id"] for h in qs.search_dense(vec, "w1", 10, doc_type="resume")] == ["b"]
    assert [h.payload["doc_id"] for h in qs.search_dense(vec, "w1", 10, filename="a.md")] == ["a"]
    assert [h.payload["doc_id"] for h in qs.search_dense(vec, "w1", 10, doc_ids=["b"])] == ["b"]
    assert qs.search_dense(vec, "nobody", 10) == []


def test_workspace_is_mandatory(qs):
    with pytest.raises(ValueError):
        qs.search_dense([1.0, 0.0, 0.0, 0.0], "", 5)
    with pytest.raises(ValueError):
        qs.iter_chunks(" ", "a")


def test_iter_chunks_is_ordered_and_delete_by_doc(qs):
    qs.upsert([_pt(3, "w", "a", 2), _pt(1, "w", "a", 0), _pt(2, "w", "a", 1), _pt(4, "w", "b", 0)])
    assert [h.payload["seq"] for h in qs.iter_chunks("w", "a")] == [0, 1, 2]
    assert qs.count() == 4
    qs.delete_by_doc("a")
    assert qs.count() == 1
    assert qs.iter_chunks("w", "a") == []


def test_empty_sparse_query_returns_nothing(qs):
    qs.upsert([_pt(1, "w", "a", 0, ([1], [1.0]))])
    assert qs.search_sparse([], [], "w", 5) == []


def test_reopening_a_stale_index_warns(tmp_path, caplog):
    """An index created without IDF (older versions) triggers a clear warning."""
    from qdrant_client import QdrantClient, models as qm

    path = str(tmp_path / "old")
    old = QdrantClient(path=path)
    old.create_collection("t", vectors_config={"dense": qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)},
                          sparse_vectors_config={"sparse": qm.SparseVectorParams()})
    old.close()
    with caplog.at_level("WARNING", logger="docmind.store"):
        QdrantStore(path, "t", DIM)
    assert "IDF" in caplog.text
