"""Two-phase ordering, persistent resume, provider failures and record isolation.

Every provider boundary is replaced. SQLite and LangGraph are real.
"""
from unittest.mock import Mock

import pytest

from src.backend import config, db, services
from src.backend.graphs import ingest


def cv(name, text=None):
    return {"cv_id": name, "file_name": name + ".txt", "file_path": "",
            "raw_text": text or name + " RAW_PRIVATE_BODY a@example.com"}


def profile(name="A"):
    return {"full_name": name, "role": "Developer", "skills": ["Python"],
            "years_experience": 4, "summary": "Builds APIs"}


def test_all_extractions_cached_before_any_embedding_and_batches_are_bulk(model_doubles, monkeypatch):
    events, calls, upserts = [], [], []
    def extract(text, missing):
        assert "a@example.com" not in text
        events.append("extract")
        return profile(), "gemini"
    def embed(texts):
        assert events.count("extract") == 23
        with db.connect() as conn:
            assert conn.execute("SELECT count(*) FROM extraction_cache").fetchone()[0] == 23
        assert all("RAW_PRIVATE_BODY" not in text and len(text.split()) < 300 for text in texts)
        calls.append(len(texts))
        return [[1., 0., 0.] for _ in texts]
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "embed_documents_with_backoff", embed)
    monkeypatch.setattr(services, "upsert_vectors", lambda ids, vectors: upserts.append(len(ids)))
    rows = ingest.run_ingest([cv(str(n)) for n in range(23)])
    assert all(row.get("status") == "ready" for row in rows)
    assert calls == upserts == [20, 3]
    assert all(db.get_cv(str(n)).get("status") == "ready" for n in range(23))


def test_embedding_failure_resumes_from_disk_cache_then_skips_ready(model_doubles, monkeypatch):
    extract = Mock(return_value=(profile(), "ollama"))
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "_sleep", lambda _: None)
    embedding = Mock()
    embedding.embed_documents.side_effect = services.GoogleRateLimitError("quota")
    monkeypatch.setattr(services, "embeddings", lambda: embedding)
    files = [cv("a"), cv("b")]
    assert all(row.get("status") == "failed" for row in ingest.run_ingest(files))
    assert embedding.embed_documents.call_count == 4
    assert extract.call_count == 2
    # New compiled graphs use SQLite cache even though the failed rows aren't ready.
    embedding.embed_documents.side_effect = None
    embedding.embed_documents.return_value = [[1., 0., 0.], [1., 0., 0.]]
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    assert all(row.get("status") == "ready" for row in ingest.run_ingest(files))
    assert extract.call_count == 2
    monkeypatch.setattr(services, "has_vector", lambda _: True)
    assert all(row.get("already_ready") for row in ingest.run_ingest(files))
    assert embedding.embed_documents.call_count == 5


def test_changed_input_invalidates_extraction_cache(model_doubles, monkeypatch):
    extract = Mock(return_value=(profile(), "gemini"))
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    monkeypatch.setattr(services, "has_vector", lambda _: True)
    ingest.run_ingest([cv("a", "First CV")])
    ingest.run_ingest([cv("a", "Changed CV")])
    assert extract.call_count == 2


@pytest.mark.parametrize("local_error", [False, True])
def test_failed_or_incomplete_local_extraction_never_retries_gemini(model_doubles, monkeypatch, local_error):
    extract = Mock(side_effect=services.LocalModelError("offline")) if local_error else Mock(
        return_value=({"full_name": "A", "skills": []}, "ollama"))
    monkeypatch.setattr(services, "extract_cv", extract)
    assert ingest.run_ingest([cv("a")])[0].get("status") == "failed"
    assert extract.call_count == 1


def test_one_bad_record_and_cache_write_do_not_crash_other_records(model_doubles, monkeypatch):
    def extract(text, missing):
        if text.startswith("bad"):
            raise ValueError("bad JSON")
        return profile(), "gemini"
    save = db.save_extraction
    def cache(cv_id, *args):
        if cv_id == "cache-error":
            raise OSError("disk error")
        save(cv_id, *args)
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(db, "save_extraction", cache)
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    rows = ingest.run_ingest([cv(name) for name in ("bad", "cache-error", "good")])
    assert [row.get("status") for row in rows] == ["failed", "failed", "ready"]


def test_bulk_upsert_failure_reuses_vectors_and_isolates_single_record(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "extract_cv", lambda *args: (profile(), "gemini"))
    embed = Mock(return_value=[[1., 0., 0.] for _ in range(3)])
    monkeypatch.setattr(services, "embed_documents_with_backoff", embed)
    def upsert(ids, vectors):
        if len(ids) > 1 or ids == ["bad"]:
            raise ValueError("storage rejected record")
    monkeypatch.setattr(services, "upsert_vectors", upsert)
    rows = ingest.run_ingest([cv(name) for name in ("a", "bad", "c")])
    assert [row.get("status") for row in rows] == ["ready", "failed", "ready"]
    embed.assert_called_once()


def test_ready_checkpoint_commits_before_next_record(model_doubles, monkeypatch):
    monkeypatch.setattr(services, "extract_cv", lambda *args: (profile(), "gemini"))
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    save, committed = db.save_cv, []
    def save_cv(cv_id, file_name, status, **kwargs):
        if status == "ready":
            assert all(db.get_cv(name).get("status") == "ready" for name in committed)
            if cv_id == "b":
                raise OSError("one record failed to commit")
            save(cv_id, file_name, status, **kwargs)
            committed.append(cv_id)
        else:
            save(cv_id, file_name, status, **kwargs)
    monkeypatch.setattr(db, "save_cv", save_cv)
    rows = ingest.run_ingest([cv(name) for name in ("a", "b", "c")])
    assert [row.get("status") for row in rows] == ["ready", "failed", "ready"]
    assert committed == ["a", "c"]


def test_real_sdk_encodes_one_batch_request_with_independent_contents(monkeypatch):
    import json
    from unittest.mock import patch
    from google import genai
    from google.genai.types import HttpResponse
    from src.backend.gemini_embeddings import GeminiEmbeddings
    # This test replaces the HTTP boundary: do not inherit host proxy settings.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    client = genai.Client(api_key="test-not-a-real-key", vertexai=False)
    response = HttpResponse(headers={}, body=json.dumps({
        "embeddings": [{"values": [3., 4., 0.]}, {"values": [0., 0., 1.]}]}))
    try:
        with patch.object(client.models._api_client, "request", return_value=response) as request:
            vectors = GeminiEmbeddings(client, "gemini-embedding-2", 3).embed_documents(["a", "b"])
        request.assert_called_once()
        method, path, body, _ = request.call_args.args
        assert method == "post" and path == "models/gemini-embedding-2:batchEmbedContents"
        assert [entry.get("content", {}).get("parts", [])[0].get("text") for entry in body.get("requests", [])] == [
            "title: none | text: a", "title: none | text: b"]
        assert all(entry.get("outputDimensionality") == 3 for entry in body.get("requests", []))
        assert vectors == [[.6, .8, 0.], [0., 0., 1.]]
    finally:
        client.close()


def test_shared_pacing_allows_one_atomic_claim_across_connections(isolated):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        waits = list(pool.map(lambda _: db.claim_cloud_slot("fixture", 5.0, 100.0), range(8)))
    assert waits.count(0.) == 1 and waits.count(5.) == 7
    assert db.claim_cloud_slot("fixture", 5.0, 104.0) == 1.
    assert db.claim_cloud_slot("fixture", 5.0, 105.0) == 0.


def test_pacing_is_shared_with_a_fresh_process(isolated):
    import json
    import os
    import subprocess
    import sys
    assert db.claim_cloud_slot("fixture", 5., 100.) == 0.
    code = "from src.backend import db; import json; print(json.dumps(db.claim_cloud_slot('fixture',5.,100.)))"
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "APP_DB_PATH": str(isolated / "app.db")},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == 5.


def test_shared_pacer_rechecks_slot_after_waking(isolated, monkeypatch):
    times = iter([100., 101., 105.])
    from src.backend import cloud_pacing
    monkeypatch.setattr(cloud_pacing.time, "time", lambda: next(times))
    waits = []
    monkeypatch.setattr(services, "_sleep", waits.append)
    limiter = services.SharedPacingLimiter()
    assert limiter.acquire() == 0.
    assert limiter.acquire() == 4.
    assert waits == [4.]


def test_two_ingests_in_one_process_do_not_overlap_phases(model_doubles, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    events = []
    def extract(text, missing):
        events.append("extract")
        return profile(), "gemini"
    def embed(texts):
        events.append("embed")
        return [[1., 0., 0.] for _ in texts]
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "embed_documents_with_backoff", embed)
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: ingest.run_ingest([cv(name)]), ("a", "b")))
    assert events == ["extract", "embed", "extract", "embed"]
    assert all(rows[0].get("status") == "ready" for rows in results)


def test_pipeline_file_lock_prevents_interleaving_across_processes(isolated):
    import os
    import subprocess
    import sys
    code = """
import sys
from src.backend.graphs.ingest import pipeline_file_lock
print('waiting', flush=True)
with pipeline_file_lock():
    print('entered', flush=True)
"""
    with ingest.pipeline_file_lock():
        child = subprocess.Popen([sys.executable, "-u", "-c", code],
            env={**os.environ, "APP_DB_PATH": str(isolated / "app.db")},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "waiting"
            with pytest.raises(subprocess.TimeoutExpired):
                child.communicate(timeout=.25)
        except BaseException:
            child.kill()
            child.communicate()
            raise
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "entered"


def test_preprocessing_masks_local_phones_without_losing_experience_dates():
    from src.backend.text_cleaning import prepare_llm_input
    text = prepare_llm_input("Python 4 years, 2022-2026. 0901234567; SĐT: 091 234 5678", 8000)
    assert "0901234567" not in text and "091 234 5678" not in text
    assert text.count("[REDACTED_PHONE]") == 2
    assert "4 years, 2022-2026" in text


def test_cli_model_check_uses_same_application_database(isolated, monkeypatch):
    from pathlib import Path
    from scripts.check_gemini_model import pace_model_check
    from src.backend import cloud_pacing
    monkeypatch.setenv("APP_DB_PATH", str(isolated / "app.db"))
    monkeypatch.setenv("GEMINI_MAX_RPM", "12")
    wait = Mock()
    monkeypatch.setattr(cloud_pacing, "wait_for_slot", wait)
    pace_model_check()
    wait.assert_called_once_with(Path(isolated / "app.db"), "gemini-3.5-flash-lite", 12)


@pytest.mark.parametrize("bad", [None, [], "invalid", {"cv_id": "bad", "extracted": []},
                                  {"cv_id": "bad", "extracted": {"full_name": ["A"], "skills": ["Python"]}}])
def test_malformed_prepared_record_fails_without_losing_next_cv(model_doubles, monkeypatch, bad):
    upsert = Mock()
    monkeypatch.setattr(services, "upsert_vectors", upsert)
    rows = ingest.save_and_index_batch({"prepared": [bad, {**cv("good"), "extracted": profile()}]}).get("processed")
    assert [row.get("status") for row in rows] == ["failed", "ready"]
    assert db.get_cv("good").get("status") == "ready"
    if isinstance(bad, dict) and bad.get("cv_id"):
        assert db.get_cv("bad").get("status") == "failed"
    upsert.assert_called_once_with(["good"], [[1., 0., 0.]])


def test_malformed_input_records_preserve_order_and_do_not_abort_batch(model_doubles, monkeypatch):
    extract = Mock(return_value=(profile(), "gemini"))
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "upsert_vectors", lambda *args: None)
    rows = ingest.run_ingest([None, [], {"cv_id": ["bad-id"]}, cv("good")])
    assert [row.get("status") for row in rows] == ["failed", "failed", "failed", "ready"]
    assert db.get_cv("good").get("status") == "ready"
    assert db.get_cv("") is None
    extract.assert_called_once()
