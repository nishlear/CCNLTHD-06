"""SQLite app.db (design 11.3.4)."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import APP_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS cv (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    raw_text TEXT,
    full_name TEXT,
    email TEXT,
    phone TEXT,
    years_experience REAL,
    skills TEXT NOT NULL DEFAULT '[]',
    education TEXT,
    summary TEXT,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    criteria TEXT,
    top_k INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id TEXT NOT NULL REFERENCES search(id) ON DELETE CASCADE,
    cv_id TEXT NOT NULL,
    score REAL NOT NULL,
    matched_skills TEXT NOT NULL,
    missing_skills TEXT NOT NULL,
    reasons TEXT,
    rank INTEGER NOT NULL,
    approved INTEGER
);
CREATE TABLE IF NOT EXISTS interview_question (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_result_id INTEGER NOT NULL REFERENCES search_result(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    question TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(APP_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---- CV ----

def cv_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["skills"] = json.loads(d["skills"] or "[]")
    return d


def save_cv(cv_id: str, file_name: str, status: str, *, raw_text: str | None = None,
            extracted: dict | None = None, error: str | None = None) -> None:
    e = extracted or {}
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO cv (id, file_name, raw_text, full_name, email, phone,
               years_experience, skills, education, summary, status, error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cv_id, file_name, raw_text, e.get("full_name"), e.get("email"), e.get("phone"),
             e.get("years_experience"), json.dumps(e.get("skills", [])), e.get("education"),
             e.get("summary"), status, error, now()),
        )


def list_cvs(status: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM cv", ()
    if status:
        sql, args = sql + " WHERE status = ?", (status,)
    with connect() as conn:
        return [cv_row_to_dict(r) for r in conn.execute(sql + " ORDER BY created_at DESC", args)]


def get_cv(cv_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM cv WHERE id = ?", (cv_id,)).fetchone()
    return cv_row_to_dict(row) if row else None


def delete_cv(cv_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM cv WHERE id = ?", (cv_id,)).rowcount > 0


# ---- Search ----

def create_search(search_id: str, query: str, top_k: int) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO search (id, query, top_k, status, created_at) VALUES (?,?,?,?,?)",
            (search_id, query, top_k, "running", now()),
        )


def update_search(search_id: str, *, status: str, criteria: dict | None = None) -> None:
    with connect() as conn:
        if criteria is not None:
            conn.execute("UPDATE search SET status = ?, criteria = ? WHERE id = ?",
                         (status, json.dumps(criteria), search_id))
        else:
            conn.execute("UPDATE search SET status = ? WHERE id = ?", (status, search_id))


def get_search(search_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM search WHERE id = ?", (search_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["criteria"] = json.loads(d["criteria"]) if d["criteria"] else None
    return d


def list_searches() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, query, top_k, status, created_at FROM search ORDER BY created_at DESC")
        return [dict(r) for r in rows]


def save_shortlist(search_id: str, shortlist: list[dict]) -> None:
    """shortlist items: cv_id, score, matched_skills, missing_skills, reasons, rank."""
    with connect() as conn:
        conn.execute("DELETE FROM search_result WHERE search_id = ?", (search_id,))
        conn.executemany(
            """INSERT INTO search_result (search_id, cv_id, score, matched_skills,
               missing_skills, reasons, rank) VALUES (?,?,?,?,?,?,?)""",
            [(search_id, s["cv_id"], s["score"], json.dumps(s["matched_skills"]),
              json.dumps(s["missing_skills"]), s.get("reasons"), s["rank"]) for s in shortlist],
        )


def get_shortlist(search_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT sr.*, cv.full_name FROM search_result sr
               LEFT JOIN cv ON cv.id = sr.cv_id
               WHERE sr.search_id = ? ORDER BY sr.rank""",
            (search_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["matched_skills"] = json.loads(d["matched_skills"])
        d["missing_skills"] = json.loads(d["missing_skills"])
        d["approved"] = None if d["approved"] is None else bool(d["approved"])
        out.append(d)
    return out


def save_review(search_id: str, approved_ids: list[str], questions: list[dict]) -> None:
    with connect() as conn:
        conn.execute("UPDATE search_result SET approved = (cv_id IN (SELECT value FROM json_each(?)))"
                     " WHERE search_id = ?", (json.dumps(approved_ids), search_id))
        result_ids = {
            r["cv_id"]: r["id"] for r in conn.execute(
                "SELECT id, cv_id FROM search_result WHERE search_id = ?", (search_id,))
        }
        conn.executemany(
            "INSERT INTO interview_question (search_result_id, category, question) VALUES (?,?,?)",
            [(result_ids[q["cv_id"]], q["category"], q["question"])
             for q in questions if q["cv_id"] in result_ids],
        )


def get_questions(search_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT sr.cv_id, iq.category, iq.question FROM interview_question iq
               JOIN search_result sr ON sr.id = iq.search_result_id
               WHERE sr.search_id = ? ORDER BY sr.rank, iq.id""",
            (search_id,),
        )
        return [dict(r) for r in rows]
