"""Draft candidate golden Q&A pairs from indexed chunks with the local LLM (for MANUAL review).

    python -m eval.make_golden --workspace default --n 12 --out eval/golden.candidates.jsonl

Each candidate's ``expected_snippet`` is verified to be an exact substring of its chunk;
all rows carry ``"needs_review": true``. Review/edit, then move good rows into golden.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="default")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "eval" / "golden.candidates.jsonl"))
    args = ap.parse_args()

    from app import config, db
    from app.llm import client as llm
    from app.retrieval.store import get_store

    config.ensure_dirs()
    db.init_db()
    docs = [d for d in db.list_docs(args.workspace, "done") if d["doc_type"] != "spreadsheet"]
    if not docs:
        sys.exit(f"No indexed text documents in workspace '{args.workspace}'. Upload some first.")
    pool = []
    for d in docs:
        for h in get_store().iter_chunks(args.workspace, d["doc_id"]):
            if len(h.payload.get("text", "")) >= 150 and h.payload.get("section") != "Extracted metadata":
                pool.append((d, h.payload["text"]))
    random.Random(args.seed).shuffle(pool)
    rows, tried = [], 0
    for d, text in pool:
        if len(rows) >= args.n or tried >= args.n * 3:
            break
        tried += 1
        try:
            out = llm.generate_json(
                f"PASSAGE:\n{text[:1500]}\n\nWrite ONE specific question that this passage answers, and copy the "
                'shortest exact phrase (under 100 characters) from the passage that answers it. Reply as JSON: '
                '{"question": "...", "snippet": "..."}',
                system="You write evaluation questions. Output only JSON.", num_predict=150)
        except llm.LLMError as e:
            sys.exit(f"LLM unavailable: {e}")
        q, snip = str((out or {}).get("question", "")).strip(), _norm(str((out or {}).get("snippet", "")))
        if len(q) < 10 or not snip or snip.lower() not in _norm(text).lower():
            continue
        rows.append({"question": q, "expected_file": d["filename"], "expected_snippet": snip,
                     "doc_type": d["doc_type"], "needs_review": True})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} candidate(s) to {args.out} - review them before use.")


if __name__ == "__main__":
    main()
