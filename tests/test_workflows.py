import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.backend import config, db, services
from src.backend.graphs import ingest, search


def fail(*args, **kwargs):
    raise RuntimeError("Injected failure")


def test_report_shortlist_and_real_interrupt(candidates):
    result = search.start_search("s", "Python FastAPI", 2)
    assert result["paused"] is True
    assert [r["score"] for r in result["shortlist"]] == [1.0, .52]
    with search.persistent_graph() as graph:
        snapshot = graph.get_state(search.graph_config("s"))
        assert snapshot.next == ("human_review",)
        assert snapshot.values["approval_status"] == "pending"
        assert snapshot.values["questions"] == []
    questions = search.resume_search("s", ["cv-a", "cv-a", "outside"])
    assert len(questions) == 3
    assert {q["cv_id"] for q in questions} == {"cv-a"}
    assert search.resume_search("s", ["cv-a"]) == questions
    with pytest.raises(ValueError):
        search.resume_search("s", ["cv-b"])


def test_empty_review_and_thread_isolation(candidates):
    search.start_search("first", "Python", 2)
    search.start_search("second", "Python", 2)
    assert search.resume_search("first", []) == []
    with search.persistent_graph() as graph:
        assert graph.get_state(search.graph_config("first")).values["approval_status"] == "rejected"
        assert graph.get_state(search.graph_config("second")).next == ("human_review",)


def test_no_candidates_ends_without_review(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "vector_search", lambda q, k: [])
    out = search.start_search("empty", "query", 10)
    assert out["shortlist"] == [] and out["paused"] is False


def test_below_threshold_skips_explanation(candidates, monkeypatch):
    monkeypatch.setattr(services, "parse_query", lambda q: {"skills": ["Java"], "min_years": 10})
    monkeypatch.setattr(services, "explain_match", fail)
    out = search.start_search("low", "Java", 2)
    assert out["paused"] and out["explanations"] == []
    assert all(r["score"] < .5 for r in out["shortlist"])


def test_invalid_criteria_uses_semantic_fallback(candidates, monkeypatch):
    monkeypatch.setattr(services, "parse_query", lambda q: {"skills": 42})
    out = search.start_search("fallback", "query", 2)
    assert out["criteria"]["keywords"] == ["query"]
    assert out["criteria"]["skills"] == []


def test_stale_cv_and_explanation_failure_are_isolated(candidates, monkeypatch):
    monkeypatch.setattr(services, "vector_search", lambda q, k: [
        {"cv_id": "stale", "similarity": 1}, {"cv_id": "cv-a", "similarity": 1}])
    monkeypatch.setattr(services, "explain_match", fail)
    out = search.start_search("stale", "query", 2)
    assert [r["cv_id"] for r in out["shortlist"]] == ["cv-a"]
    assert out["explanations"] == [{"cv_id": "cv-a", "reasons": None}]


def test_question_failure_does_not_abort_other_cv(candidates, monkeypatch):
    def generate(cv_id, cv, criteria):
        if cv_id == "cv-a":
            raise RuntimeError("Injected question error")
        return [{"cv_id": cv_id, "category": "technical", "question": "Python?"}]
    monkeypatch.setattr(services, "generate_questions", generate)
    search.start_search("questions", "query", 2)
    assert [q["cv_id"] for q in search.resume_search("questions", ["cv-a", "cv-b"])] == ["cv-b"]


def test_api_review_rejects_repeated_requests(client, candidates):
    response = client.post("/searches", json={"query": "Python", "top_k": 2})
    assert response.status_code == 200
    search_id = response.json()["id"]
    review = client.post(f"/searches/{search_id}/review", json={"approved_cv_ids": ["cv-a", "outside"]})
    assert review.status_code == 200 and len(review.json()["questions"]) == 3
    assert client.post(f"/searches/{search_id}/review", json={"approved_cv_ids": []}).status_code == 409
    detail = client.get(f"/searches/{search_id}").json()
    assert detail["approved_cv_ids"] == ["cv-a"] and detail["status"] == "completed"


def test_api_retrieval_error_marks_failed(client, monkeypatch):
    monkeypatch.setattr(services, "vector_search", fail)
    assert client.post("/searches", json={"query": "Python"}).status_code == 500
    assert db.list_searches()[0]["status"] == "failed"


def test_missing_key_is_clear_and_health_remains_available(isolated, monkeypatch):
    from fastapi.testclient import TestClient
    from src.backend.main import app
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/searches", json={"query": "Python"}).status_code == 503
        assert db.list_searches() == []


@pytest.mark.parametrize("query", ["", "   ", "x" * 3001])
def test_query_validation(client, query):
    assert client.post("/searches", json={"query": query}).status_code == 422


def test_ingest_retry_and_partial_failure(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "pdf_to_text", lambda path: path)
    calls = {}
    def extract(text, missing):
        calls[text] = calls.get(text, 0) + 1
        if text == "good" and calls[text] == 2:
            assert missing == ["skills"]
            return {"full_name": "Candidate", "skills": ["Python"]}
        return {"full_name": "Candidate", "skills": []}
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "index_cv", lambda *a: None)
    out = ingest.run_ingest([{"cv_id": name, "file_name": name + ".pdf", "file_path": name}
                             for name in ("good", "bad")])
    assert [r["status"] for r in out] == ["ready", "failed"]
    assert calls == {"good": 2, "bad": 2}
    assert "skills" in db.get_cv("bad")["error"]


def test_pdf_and_index_errors_are_per_file(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "pdf_to_text", fail)
    file = {"cv_id": "bad", "file_name": "bad.pdf", "file_path": "bad.pdf"}
    assert ingest.run_ingest([file])[0]["status"] == "failed"
    monkeypatch.setattr(services, "pdf_to_text", lambda path: "CV text")
    monkeypatch.setattr(services, "extract_cv", lambda *args: {"full_name": "Candidate", "skills": ["Python"]})
    monkeypatch.setattr(services, "index_cv", fail)
    assert ingest.run_ingest([file])[0]["status"] == "failed"
    assert db.get_cv("bad")["skills"] == ["Python"]


def test_upload_rejection_cleans_staged_files(client):
    response = client.post("/cvs", files=[("files", ("a.pdf", b"%PDF-fixture", "application/pdf")),
                                         ("files", ("b.txt", b"text", "text/plain"))])
    assert response.status_code == 400
    assert list(config.UPLOAD_DIR.iterdir()) == []


def test_resume_through_api_in_a_new_process(client, candidates, isolated):
    search_id = client.post("/searches", json={"query": "Python"}).json()["id"]
    code = """
import json
from fastapi.testclient import TestClient
from src.backend import services
from src.backend.main import app
from src.backend.graphs.search import persistent_graph, graph_config

def unexpected(*args, **kwargs):
    raise AssertionError('A completed step was repeated')
services.parse_query = unexpected
services.vector_search = unexpected
services.generate_questions = lambda cv_id, cv, criteria: [dict(cv_id=cv_id, category='technical', question='Restart?')]
with TestClient(app) as client:
    response = client.post('/searches/SEARCH_ID/review', json={'approved_cv_ids': ['cv-a']})
    assert response.status_code == 200, response.text
    assert response.json()['questions'][0]['cv_id'] == 'cv-a'
with persistent_graph() as graph:
    snapshot = graph.get_state(graph_config('SEARCH_ID'))
    assert not snapshot.next
    assert snapshot.values['approval_status'] == 'approved'
print(json.dumps({'resumed': True}))
""".replace("SEARCH_ID", search_id)
    env = {**os.environ, "DATA_DIR": str(isolated), "APP_DB_PATH": str(isolated / "app.db"),
           "CHECKPOINT_DB_PATH": str(isolated / "checkpoints.db")}
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resumed"]
    assert db.get_search(search_id)["status"] == "completed"


def test_anonymous_corpus_text_does_not_require_pdf_or_fabricated_name(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "pdf_to_text", fail)
    monkeypatch.setattr(services, "extract_cv", lambda text, missing: {"full_name": "", "skills": ["Python"]})
    monkeypatch.setattr(services, "index_cv", lambda *args: None)
    rows = ingest.run_ingest([{"cv_id": "djinni-a", "file_name": "a.txt", "file_path": "",
                               "raw_text": "Real anonymous candidate text", "display_name": "Anonymous candidate a"}])
    assert rows[0]["status"] == "ready"
    assert db.get_cv("djinni-a")["full_name"] == "Anonymous candidate a"
