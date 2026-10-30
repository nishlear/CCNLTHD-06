"""Bounded corpus operations shared by the GUI and the existing ingest pipeline.

Jobs and cursors live on disk. No shell commands or client-supplied paths are run.
Only an explicit ingest job crosses the model boundary.
"""
import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config, db, services
from .graphs.ingest import run_ingest
from scripts.run_dataset import prepare_ingest_item

ACTIVE = ("queued", "running", "stopping")


class JobConflict(Exception):
    pass


class Stopped(Exception):
    pass


@contextmanager
def job_connection():
    path = db.APP_DB_PATH.with_name("corpus_jobs.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS job (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, lease REAL NOT NULL, body TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_corpus_job ON job ((1))
                WHERE status IN ('queued','running','stopping');
            CREATE TABLE IF NOT EXISTS cursor (
                corpus TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, next_record INTEGER NOT NULL);
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _expire(conn):
    for row in conn.execute("SELECT * FROM job WHERE status IN ('queued','running','stopping') AND lease < ?", (time.time(),)):
        body = json.loads(row["body"])
        body.update(status="interrupted", phase="Phiên xử lý đã dừng. Có thể chạy lô tiếp theo để tiếp tục.")
        conn.execute("UPDATE job SET status='interrupted', body=? WHERE id=?", (json.dumps(body), row["id"]))


def update_job(job_id, **fields):
    with job_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT body,status FROM job WHERE id=?", (job_id,)).fetchone()
        body = json.loads(row["body"])
        # A concurrent stop request must survive a worker's progress update.
        if row["status"] == "stopping" and fields.get("status") == "running":
            fields["status"] = "stopping"
        body.update(fields, updated_at=db.now())
        conn.execute("UPDATE job SET status=?,body=?,lease=? WHERE id=?",
                     (body["status"], json.dumps(body, ensure_ascii=False), time.time() + 60, job_id))
        return body


def list_jobs():
    with job_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _expire(conn)
        return [json.loads(row["body"]) for row in conn.execute("SELECT body FROM job ORDER BY rowid DESC LIMIT 30")]


def get_job(job_id):
    with job_connection() as conn:
        row = conn.execute("SELECT body FROM job WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["body"]) if row else None


def dataset_dir():
    return config.DATA_DIR / "datasets" / "djinni"


def resolve_corpus(corpus_id):
    if not corpus_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", corpus_id):
        raise ValueError("Hãy chọn một tập hồ sơ có trong danh sách.")
    root = dataset_dir().resolve()
    path = (root / f"{corpus_id}.jsonl").resolve()
    manifest_path = (root / f"{corpus_id}.manifest.json").resolve()
    if path.parent != root or manifest_path.parent != root or not path.is_file() or not manifest_path.is_file():
        raise ValueError("Không tìm thấy tập hồ sơ và manifest tương ứng.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("selected"), int) or manifest["selected"] < 1:
        raise ValueError("Manifest không có số hồ sơ hợp lệ.")
    return path, manifest


def read_cursor(corpus_id, fingerprint):
    with job_connection() as conn:
        row = conn.execute("SELECT * FROM cursor WHERE corpus=?", (corpus_id,)).fetchone()
        return row["next_record"] if row and row["fingerprint"] == fingerprint else 0


def save_cursor(corpus_id, fingerprint, position):
    with job_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO cursor VALUES (?,?,?)", (corpus_id, fingerprint, position))


def overview():
    corpora = []
    for path in sorted(dataset_dir().glob("*.manifest.json")):
        corpus_id = path.name.removesuffix(".manifest.json")
        try:
            data, manifest = resolve_corpus(corpus_id)
            corpora.append({"id": corpus_id, "records": manifest["selected"], "bytes": data.stat().st_size,
                            "next_record": read_cursor(corpus_id, manifest.get("output_sha256", "")),
                            "dataset": manifest.get("dataset", "Djinni"), "seed": manifest.get("seed")})
        except (OSError, ValueError, TypeError):
            continue
    with db.connect() as conn:
        counts = {row["status"]: row["n"] for row in conn.execute("SELECT status,count(*) n FROM cv GROUP BY status")}
    return {"corpora": corpora, "source_available": (dataset_dir() / "train.parquet").is_file(),
            "application_counts": counts, "jobs": list_jobs()}


def preview(corpus_id, offset, limit):
    path, manifest = resolve_corpus(corpus_id)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if len(rows) >= limit:
                break
            try:
                raw = json.loads(line)
                payload, _ = prepare_ingest_item(raw)
                stored = db.get_cv(payload["cv_id"]) or {}
                rows.append({"record": index + 1, "cv_id": payload["cv_id"],
                             "name": payload.get("display_name") or "Hồ sơ ứng viên",
                             "position": raw.get("position", ""), "status": stored.get("status", "not_processed"),
                             "excerpt": payload["raw_text"][:350]})
            except (ValueError, TypeError, KeyError, AttributeError):
                rows.append({"record": index + 1, "status": "invalid", "name": "Dòng dữ liệu không hợp lệ", "excerpt": ""})
    return {"rows": rows, "total": manifest["selected"], "offset": offset, "limit": limit}


def start_job(params):
    action = params["action"]
    if action == "prepare":
        if not (dataset_dir() / "train.parquet").is_file():
            raise ValueError("Chưa có dữ liệu nguồn train.parquet trong thư mục corpus.")
    else:
        resolve_corpus(params.get("corpus_id"))
    if action == "ingest":
        services.require_gemini()
    job_id = uuid.uuid4().hex
    body = {"id": job_id, "action": action, "params": params, "status": "queued", "phase": "Đang chuẩn bị",
            "created_at": db.now(), "updated_at": db.now(), "processed": 0, "ready": 0, "failed": 0,
            "skipped": 0, "scanned": 0, "total": params.get("limit", 0), "elapsed_seconds": 0,
            "result": None, "errors": []}
    with job_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _expire(conn)
        if conn.execute("SELECT 1 FROM job WHERE status IN ('queued','running','stopping')").fetchone():
            raise JobConflict("Một tác vụ corpus đang chạy. Hãy đợi hoặc yêu cầu dừng trước khi chạy lượt khác.")
        conn.execute("INSERT INTO job VALUES (?,?,?,?)", (job_id, "queued", time.time() + 60, json.dumps(body)))
    threading.Thread(target=_worker, args=(job_id,), daemon=True, name=f"corpus-{job_id[:8]}").start()
    return body


def stop_job(job_id):
    with job_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT body,status FROM job WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        if row["status"] not in ACTIVE:
            raise JobConflict("Tác vụ đã kết thúc.")
        job = json.loads(row["body"])
        job.update(status="stopping", phase="Sẽ dừng sau lô đang xử lý; kết quả đã lưu được giữ lại.")
        conn.execute("UPDATE job SET status='stopping',body=? WHERE id=?", (json.dumps(job), job_id))
        return job


def _check_stop(job_id):
    if get_job(job_id)["status"] == "stopping":
        raise Stopped()


def _verify(job_id, path, manifest):
    digest = hashlib.sha256()
    size, read = path.stat().st_size, 0
    update_job(job_id, phase="Kiểm tra tính toàn vẹn của dữ liệu", total=size, progress=0, unit="byte")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            _check_stop(job_id)
            digest.update(chunk)
            read += len(chunk)
            update_job(job_id, progress=read)
    if digest.hexdigest() != manifest.get("output_sha256"):
        raise ValueError("Checksum không khớp manifest. Không xử lý tập dữ liệu đã thay đổi.")
    return {"checksum_matches": True, "records": manifest["selected"], "sha256": digest.hexdigest()}


def _ingest(job_id, params, path, manifest):
    fingerprint = manifest["output_sha256"]
    start = read_cursor(params["corpus_id"], fingerprint) if params["mode"] == "next" else params["offset"]
    counts = {"processed": 0, "ready": 0, "failed": 0, "skipped": 0, "scanned": 0}
    errors, batch, position = [], [], start
    update_job(job_id, total=params["limit"], progress=0, unit="hồ sơ", start_record=start + 1,
               phase="Chọn hồ sơ chưa hoàn tất")

    def flush():
        nonlocal batch
        _check_stop(job_id)
        update_job(job_id, phase=f"Trích xuất → lưu cache → tạo vector ({len(batch)} hồ sơ)")
        outcomes = run_ingest(batch)
        counts["processed"] += len(batch)
        for row in outcomes:
            counts["ready" if row["status"] == "ready" else "failed"] += 1
            if row["status"] != "ready":
                errors.append({"cv_id": row["cv_id"], "message": row.get("error")})
        if params["mode"] == "next":
            save_cursor(params["corpus_id"], fingerprint, position)
        update_job(job_id, **counts, progress=counts["processed"], next_record=position, errors=errors[-50:])
        batch = []

    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < start:
                continue
            _check_stop(job_id)
            if params["mode"] == "range" and index >= start + params["limit"]:
                break
            if counts["processed"] + len(batch) >= params["limit"]:
                break
            position = index + 1
            counts["scanned"] += 1
            try:
                raw = json.loads(line)
                payload, _ = prepare_ingest_item(raw)
                if not payload.get("cv_id") or not payload.get("raw_text"):
                    raise ValueError("Missing CV ID or text")
                existing = db.get_cv(payload["cv_id"]) or {}
                if (existing.get("status") == "ready" and existing.get("raw_text") == payload["raw_text"]
                        and services.has_vector(payload["cv_id"])):
                    counts["skipped"] += 1
                    continue
                batch.append(payload)
            except (ValueError, TypeError, AttributeError):
                counts["processed"] += 1
                counts["failed"] += 1
                errors.append({"record": position, "message": "Dòng dữ liệu không hợp lệ"})
            if len(batch) >= params["batch_size"]:
                flush()
            elif counts["scanned"] % 100 == 0:
                update_job(job_id, **counts, progress=counts["processed"], errors=errors[-50:])
    if batch:
        flush()
    if params["mode"] == "next":
        save_cursor(params["corpus_id"], fingerprint, position)
    update_job(job_id, **counts, progress=counts["processed"], next_record=position, errors=errors[-50:])
    return {"selected_for_processing": counts["processed"], "next_record": position,
            "end_of_corpus": position >= manifest["selected"], "note": "Hồ sơ lỗi có thể thử lại bằng chế độ chọn vị trí."}


def _worker(job_id):
    started = time.perf_counter()
    heartbeat_done = threading.Event()

    def heartbeat():
        while not heartbeat_done.wait(5):
            try:
                update_job(job_id, elapsed_seconds=round(time.perf_counter() - started, 3))
            except sqlite3.Error:
                pass

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        _check_stop(job_id)
        update_job(job_id, status="running")
        params = get_job(job_id)["params"]
        action = params["action"]
        if action == "prepare":
            from scripts.prepare_djinni import prepare_dataset
            output = dataset_dir() / f"sample-{params['sample_count']}-{job_id[:8]}.jsonl"
            def progress(n):
                _check_stop(job_id)
                update_job(job_id, phase="Lọc hồ sơ hợp lệ, loại trùng và lấy mẫu", progress=n, total=0, unit="dòng nguồn")
            result = prepare_dataset(dataset_dir() / "train.parquet", output, params["sample_count"], params["seed"], progress)
            result = {"corpus_id": output.stem, "records": result["selected"], "sha256": result["output_sha256"]}
        else:
            path, manifest = resolve_corpus(params["corpus_id"])
            result = _verify(job_id, path, manifest)
            if action == "ingest":
                result = _ingest(job_id, params, path, manifest)
            elif action == "benchmark":
                from scripts.benchmark_corpus import benchmark_dataset
                folder = config.DATA_DIR / "benchmarks" / "gui"
                def progress(n):
                    _check_stop(job_id)
                    update_job(job_id, phase="Lọc thông tin liên hệ, lập chỉ mục và đo tìm kiếm từ khóa",
                               progress=n, total=manifest["selected"], unit="hồ sơ")
                result = benchmark_dataset(path, folder / f"{job_id}.sqlite", folder / f"{job_id}.json",
                                           params["repeats"], True, progress)
        _check_stop(job_id)
        has_errors = get_job(job_id)["failed"] > 0
        update_job(job_id, status="completed_with_errors" if has_errors else "completed",
                   phase="Đã hoàn tất; có hồ sơ lỗi" if has_errors else "Đã hoàn tất",
                   result=result)
    except Stopped:
        update_job(job_id, status="stopped", phase="Đã dừng; giữ nguyên kết quả đã lưu")
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("Checksum", "Dataset SHA256", "Only ")) else f"Không thể hoàn tất tác vụ ({type(exc).__name__}). Kiểm tra dữ liệu, kết nối và cấu hình máy chủ."
        update_job(job_id, status="failed", phase=message)
    finally:
        update_job(job_id, elapsed_seconds=round(time.perf_counter() - started, 3))
        heartbeat_done.set()
