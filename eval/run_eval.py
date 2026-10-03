"""Retrieval evaluation: per-doc-type recall@4 and MRR for dense / hybrid / hybrid+rerank.

    python -m eval.run_eval                 # real models (downloaded once), needs no LLM
    python -m eval.run_eval --judge         # + local LLM-as-judge faithfulness (needs Ollama)
    python -m eval.run_eval --fake          # hashing stand-ins: smoke-tests the harness only

Writes eval/results.md. A hit = a retrieved chunk from ``expected_file`` whose text contains
``expected_snippet`` (whitespace/case-insensitive). MRR is computed over the top 10.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODES = ["dense", "hybrid", "hybrid_rerank"]
WS = "eval"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _setup(args: argparse.Namespace) -> None:
    os.environ["DOCMIND_DATA_DIR"] = str(Path(args.data_dir).resolve())
    if args.fake:
        os.environ["DOCMIND_FAKE_MODELS"] = "1"
        os.environ["STORE_BACKEND"] = "memory"
    os.environ["EXTRACT_METADATA"] = "true" if args.judge else "false"  # keep retrieval eval LLM-free


def _index_samples() -> None:
    from app import config, db
    from app.ingest import pipeline, spreadsheet
    config.ensure_dirs()
    db.init_db()
    spreadsheet.init_sheets_db()
    for d in db.list_docs(WS):
        pipeline.remove_document(d["doc_id"])
    for f in sorted((ROOT / "samples").iterdir()):
        if not f.is_file():
            continue
        fd, tmp = tempfile.mkstemp(dir=config.settings.data_dir)
        os.close(fd)
        os.unlink(tmp)
        shutil.copy(f, tmp)
        res = pipeline.register_upload(WS, f.name, Path(tmp))
        pipeline.ingest_document(res.doc_id)
        doc = db.get_doc(res.doc_id)
        print(f"  indexed {f.name:28s} -> {doc['doc_type']:12s} {doc['status']} {doc['n_chunks']} chunks"
              + (f"  ERROR: {doc['error']}" if doc["error"] else ""))


def _rank(hits, item) -> int | None:
    snippet = _norm(item["expected_snippet"])
    for i, h in enumerate(hits, start=1):
        p = h.payload
        if p.get("filename") == item["expected_file"] and snippet in _norm(p.get("text", "")):
            return i
    return None


def evaluate(golden: list[dict]) -> tuple[dict, list[float]]:
    from app.retrieval.hybrid import retrieve
    stats: dict = {m: defaultdict(lambda: {"n": 0, "hit4": 0, "rr": 0.0}) for m in MODES}
    top1_scores: list[float] = []
    for item in golden:
        for mode in MODES:
            hits = retrieve(item["question"], WS, mode=mode, top_k=10)
            r = _rank(hits, item)
            for key in (item["doc_type"], "ALL"):
                s = stats[mode][key]
                s["n"] += 1
                s["hit4"] += int(r is not None and r <= 4)
                s["rr"] += (1.0 / r) if r else 0.0
            if mode == "hybrid_rerank" and hits:
                top1_scores.append(hits[0].score)
    return stats, top1_scores


def judge(golden: list[dict]) -> float | None:
    """Local LLM-as-judge: is every claim in the answer supported by its sources? (1-5 -> 0-1)"""
    from app.llm import client as llm
    from app.retrieval import answer
    scores = []
    for item in golden:
        final = None
        for ev in answer.ask(item["question"], WS, use_cache=False):
            if ev["type"] == "final":
                final = ev
            elif ev["type"] == "error":
                print("  judge skipped:", ev["message"])
                return None
        if not final or not final["sources"]:
            continue
        ctx = "\n\n".join(s["passage"][:1200] for s in final["sources"][:4])
        out = llm.generate_json(
            f"SOURCES:\n{ctx}\n\nANSWER:\n{final['answer']}\n\nRate from 1 (unsupported / hallucinated) to 5 "
            '(every claim is supported by the SOURCES). Reply as JSON: {"score": <1-5>}',
            system="You are a strict faithfulness judge. Output only JSON.", num_predict=30)
        try:
            scores.append((float((out or {}).get("score")) - 1) / 4)
        except (TypeError, ValueError):
            continue
    return statistics.mean(scores) if scores else None


def render(stats: dict, top1: list[float], faith: float | None, fake: bool) -> str:
    types = sorted({k for m in stats.values() for k in m if k != "ALL"}) + ["ALL"]
    lines = ["# DocMind retrieval evaluation", ""]
    if fake:
        lines += ["> **Smoke run with hashing stand-ins (`--fake`). Numbers say nothing about the real models.**", ""]
    lines += ["| doc_type | n | " + " | ".join(f"{m} R@4 | {m} MRR" for m in MODES) + " |",
              "|---|---|" + "---|---|" * len(MODES)]
    for t in types:
        n = stats[MODES[0]][t]["n"]
        cells = []
        for m in MODES:
            s = stats[m][t]
            cells += [f"{s['hit4'] / s['n']:.2f}", f"{s['rr'] / s['n']:.2f}"]
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    if top1:
        lines += ["", f"Rerank top-1 score on these in-corpus questions: min {min(top1):.2f}, "
                      f"median {statistics.median(top1):.2f}. Set `RERANK_THRESHOLD` a few points below the min "
                      "(and above the scores you see for off-topic questions)."]
    if faith is not None:
        lines += ["", f"LLM-as-judge faithfulness (0-1): **{faith:.2f}**"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.md"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "eval"))
    ap.add_argument("--fake", action="store_true", help="use hashing stand-in models (offline smoke test)")
    ap.add_argument("--judge", action="store_true", help="also run the local LLM-as-judge faithfulness score")
    args = ap.parse_args()
    _setup(args)
    from app import config
    config.reload()
    golden = [json.loads(l) for l in Path(args.golden).read_text().splitlines() if l.strip()]
    print(f"Indexing samples into workspace '{WS}' ...")
    _index_samples()
    print(f"Evaluating {len(golden)} questions ...")
    stats, top1 = evaluate(golden)
    faith = judge(golden) if args.judge else None
    report = render(stats, top1, faith, args.fake)
    print("\n" + report)
    Path(args.out).write_text(report)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
