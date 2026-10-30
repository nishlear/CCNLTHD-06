"""Shared cloud admission clock; stdlib only, also used by CLI diagnostics."""
import sqlite3
import time
from pathlib import Path


def claim_slot(path: Path, name: str, interval: float, current_time: float | None = None) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS api_pacing (name TEXT PRIMARY KEY, next_at REAL NOT NULL)")
        conn.execute("BEGIN IMMEDIATE")
        current_time = time.time() if current_time is None else current_time
        row = conn.execute("SELECT next_at FROM api_pacing WHERE name=?", (name,)).fetchone()
        wait = max(0.0, row[0] - current_time) if row else 0.0
        if wait == 0:
            conn.execute("INSERT OR REPLACE INTO api_pacing VALUES (?,?)", (name, current_time + interval))
        conn.commit()
        return wait
    finally:
        conn.close()


def wait_for_slot(path: Path, name: str, rpm: int) -> None:
    if not 1 <= rpm <= 12:
        raise ValueError("Chat RPM must be 1..12")
    while True:
        wait = claim_slot(path, name, 60.0 / rpm)
        if not wait:
            return
        time.sleep(wait)
