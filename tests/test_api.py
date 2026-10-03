"""HTTP-level test through FastAPI's TestClient (skipped if fastapi/httpx are not installed)."""
import json
import time

import pytest

from tests.conftest import SAMPLES


def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient
    from app.api.main import app
    return TestClient(app)


def _sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        ev = next(l[7:] for l in block.splitlines() if l.startswith("event: "))
        data = next(l[6:] for l in block.splitlines() if l.startswith("data: "))
        out.append((ev, json.loads(data)))
    return out


def test_upload_status_ask_delete_flow():
    with _client() as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "<title>DocMind</title>" in c.get("/").text
        up = c.post("/documents", files={"file": ("hr_leave_policy.md", (SAMPLES / "hr_leave_policy.md").read_bytes())},
                    data={"workspace": "t"})
        assert up.status_code == 202
        doc_id = up.json()["doc_id"]
        for _ in range(100):
            st = c.get(f"/documents/{doc_id}/status").json()
            if st["status"] in {"done", "failed"}:
                break
            time.sleep(0.05)
        assert st["status"] == "done" and st["doc_type"] == "policy"
        dup = c.post("/documents", files={"file": ("again.md", (SAMPLES / "hr_leave_policy.md").read_bytes())},
                     data={"workspace": "t"})
        assert dup.json()["duplicate"] is True
        r = c.post("/ask", json={"question": "How many days of paid annual leave do full-time employees get?",
                                 "workspace": "t"})
        events = _sse(r.text)
        kinds = [k for k, _ in events]
        assert "token" in kinds and kinds[-1] == "final"
        assert events[-1][1]["sources"][0]["filename"] == "hr_leave_policy.md"
        assert c.get("/metrics/recent?workspace=t").json()[0]["route"] == "factual"
        assert c.post("/documents", files={"file": ("x.exe", b"abc")}, data={"workspace": "t"}).status_code == 415
        assert c.post("/ask", json={"question": "x", "doc_type": "bogus"}).status_code == 400
        assert c.delete(f"/documents/{doc_id}").status_code == 200
        assert c.get(f"/documents/{doc_id}/status").status_code == 404
        assert c.get("/documents?workspace=t").json() == []
