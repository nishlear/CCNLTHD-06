"""GUI corpus jobs use actual SQLite/LangGraph; cloud boundaries are replaced."""
import hashlib
import json
import threading
import time

import pytest

from src.backend import config, corpus, db, services


@pytest.fixture
def sample(client, isolated):
    folder = isolated / "datasets" / "djinni"
    folder.mkdir(parents=True)
    rows = [{"cv_id": f"cv-{i}", "file_name": f"cv-{i}.txt", "display_name": f"Anonymous {i}",
             "position": "Python developer", "raw_text": f"Python FastAPI engineer {i}. Contact test@example.com"}
            for i in range(8)]
    path = folder / "sample.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {"selected": len(rows), "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rows, path


def wait_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/corpus/jobs/{job_id}").json()
        if job["status"] not in corpus.ACTIVE and job["elapsed_seconds"] > 0:
            return job
        time.sleep(.01)
    pytest.fail("Corpus job did not finish")


def launch(client, **kwargs):
    response = client.post("/corpus/jobs", json={"corpus_id": "sample", **kwargs})
    assert response.status_code == 202, response.text
    return wait_job(client, response.json()["id"])


def test_catalog_preview_redaction_and_checksum(client, sample):
    catalog = client.get("/corpus").json()
    assert catalog["corpora"][0]["records"] == 8
    preview = client.get("/corpus/sample/preview?offset=2&limit=2").json()
    assert [r["record"] for r in preview["rows"]] == [3, 4]
    assert "test@example.com" not in json.dumps(preview)
    job = launch(client, action="verify")
    assert job["status"] == "completed" and job["result"]["checksum_matches"]
    report = client.get(f'/corpus/jobs/{job["id"]}/report')
    assert report.json()["id"] == job["id"]
    assert "attachment" in report.headers["content-disposition"]
    assert "test@example.com" not in report.text


def test_checksum_mismatch_does_not_call_ingest(client, sample, monkeypatch):
    sample[1].write_text(sample[1].read_text() + "\n")
    monkeypatch.setattr(corpus, "run_ingest", lambda _: pytest.fail("Must not ingest altered data"))
    job = launch(client, action="ingest")
    assert job["status"] == "failed" and "Checksum" in job["phase"]


def test_next_batches_skip_ready_then_continue_after_saved_cursor(client, sample, monkeypatch):
    indexed = {"cv-0"}
    calls = []
    payload, _ = corpus.prepare_ingest_item(sample[0][0])
    db.save_cv("cv-0", "0.txt", "ready", raw_text=payload["raw_text"], extracted={"full_name": "A", "skills": ["Python"]})
    monkeypatch.setattr(services, "has_vector", lambda cv_id: cv_id in indexed)
    def extract(text, missing):
        assert "test@example.com" not in text
        calls.append(text)
        return {"full_name": "Candidate", "skills": ["Python"]}, "gemini"
    monkeypatch.setattr(services, "extract_cv", extract)
    monkeypatch.setattr(services, "embed_documents_with_backoff", lambda texts: [[1., 0., 0.] for _ in texts])
    monkeypatch.setattr(services, "upsert_vectors", lambda ids, vectors: indexed.update(ids))
    first = launch(client, action="ingest", limit=2, batch_size=1)
    assert (first["ready"], first["skipped"], first["next_record"]) == (2, 1, 3)
    second = launch(client, action="ingest", limit=2, batch_size=1)
    assert (second["ready"], second["next_record"]) == (2, 5)
    assert len(calls) == 4 and indexed == {f"cv-{i}" for i in range(5)}
    # A range retry leaves the forward cursor alone and reuses ready vectors.
    retry = launch(client, action="ingest", limit=1, batch_size=1, mode="range", offset=1)
    assert retry["skipped"] == 1 and retry["ready"] == 0
    assert client.get("/corpus").json()["corpora"][0]["next_record"] == 5


def test_stop_preserves_finished_batch_and_disallows_overlapping_jobs(client, sample, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def ingest(items):
        entered.set()
        assert release.wait(5)
        return [{"cv_id": item["cv_id"], "status": "ready"} for item in items]
    monkeypatch.setattr(corpus, "run_ingest", ingest)
    monkeypatch.setattr(services, "has_vector", lambda _: False)
    response = client.post("/corpus/jobs", json={"action": "ingest", "corpus_id": "sample", "limit": 4, "batch_size": 2})
    job_id = response.json()["id"]
    try:
        assert entered.wait(5)
        assert client.post("/corpus/jobs", json={"action": "verify", "corpus_id": "sample"}).status_code == 409
        assert client.post(f"/corpus/jobs/{job_id}/stop").json()["status"] == "stopping"
    finally:
        release.set()
    job = wait_job(client, job_id)
    assert job["status"] == "stopped" and job["ready"] == 2
    assert client.get("/corpus").json()["corpora"][0]["next_record"] == 2
    assert client.post(f"/corpus/jobs/{job_id}/stop").status_code == 409


def test_benchmark_uses_real_fts_and_filtered_text(client, sample):
    job = launch(client, action="benchmark", repeats=2)
    assert job["status"] == "completed", job
    result = job["result"]
    assert result["records"] == 8 and len(result["queries"]) == 5
    assert result["queries"][0]["matches"] == 8
    assert result["indexed_text"] == "contact-filtered"
    assert result["embedding_requests"] == 0


def test_failed_record_is_reported_without_losing_other_results(client, sample, monkeypatch):
    monkeypatch.setattr(services, "has_vector", lambda _: False)
    def ingest(items):
        return [{"cv_id": item["cv_id"], "status": "failed" if item["cv_id"] == "cv-1" else "ready",
                 "error": "Invalid extraction" if item["cv_id"] == "cv-1" else None} for item in items]
    monkeypatch.setattr(corpus, "run_ingest", ingest)
    job = launch(client, action="ingest", limit=3, batch_size=2)
    assert job["status"] == "completed_with_errors"
    assert (job["processed"], job["ready"], job["failed"]) == (3, 2, 1)
    assert job["errors"] == [{"cv_id":"cv-1", "message":"Invalid extraction"}]
    assert job["next_record"] == 3


def test_expired_worker_is_marked_interrupted_but_history_remains(client, sample):
    job = {"id": "old", "status": "running", "phase": "old"}
    with corpus.job_connection() as conn:
        conn.execute("INSERT INTO job VALUES (?,?,?,?)", ("old", "running", 0, json.dumps(job)))
    assert client.get("/corpus").json()["jobs"][0]["status"] == "interrupted"
    assert launch(client, action="verify")["status"] == "completed"


@pytest.mark.parametrize("payload", [
    {"action": "shell", "command": "anything"},
    {"action": "ingest", "corpus_id": "sample", "limit": 51},
    {"action": "ingest", "corpus_id": "sample", "batch_size": 21},
    {"action": "verify", "corpus_id": "../sample"},
    {"action": "benchmark", "corpus_id": "sample", "repeats": 0},
])
def test_invalid_operation_or_path_is_rejected(client, sample, payload):
    assert client.post("/corpus/jobs", json=payload).status_code in (400, 422)


def test_prepare_is_a_new_sample_not_an_overwrite(client, sample, monkeypatch):
    import scripts.prepare_djinni as prepare
    source = sample[1].parent / "train.parquet"
    source.write_bytes(b"source")
    original = sample[1].read_bytes()
    def fake_prepare(source_path, output, limit, seed, progress):
        assert source_path == source and (limit, seed) == (3, 123)
        assert output != sample[1] and not output.exists()
        progress(1000)
        return {"selected": 3, "output_sha256": "abc"}
    monkeypatch.setattr(prepare, "prepare_dataset", fake_prepare)
    job = launch(client, action="prepare", sample_count=3, seed=123)
    assert job["status"] == "completed" and job["result"]["records"] == 3
    assert sample[1].read_bytes() == original


def test_shared_prepare_writes_matching_manifest_and_keeps_cli_contract(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from scripts import prepare_djinni
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"id":str(i), "CV":"Python engineer " + str(i) + " building APIs and tests for reliable applications. " * 3,
         "CV_lang":"en", "Position":"Developer"} for i in range(5)
    ]), source)
    monkeypatch.setattr(prepare_djinni, "EXPECTED_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "sample.jsonl"
    manifest = prepare_djinni.prepare_dataset(source, output, 3, 123)
    assert manifest["selected"] == 3 and manifest["eligible_unique"] == 5
    assert len(output.read_text().splitlines()) == 3
    saved = json.loads(output.with_suffix(".manifest.json").read_text())
    assert saved["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        prepare_djinni.prepare_dataset(source, output, 3, 123)
