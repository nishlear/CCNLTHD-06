"""Reproducible sampler for the CV benchmark set.

Run from repo root:  python data/benchmark/make_sample.py
Reads data/archive/Resume/Resume.csv and writes data/benchmark/cvs.jsonl.
"""
import csv
import json
import random
import re
from pathlib import Path

SEED = 42
PER_CATEGORY = 10
MIN_CHARS = 500
CATEGORIES = [
    "INFORMATION-TECHNOLOGY", "ENGINEERING", "FINANCE", "ACCOUNTANT", "BANKING",
    "CHEF", "HEALTHCARE", "DESIGNER", "DIGITAL-MEDIA", "SALES",
]

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "archive" / "Resume" / "Resume.csv"
OUT = ROOT / "data" / "benchmark" / "cvs.jsonl"


def main() -> None:
    csv.field_size_limit(10**9)
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    skipped: list[tuple[str, str, int]] = []
    with SRC.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cat = row["Category"].strip()
            if cat not in by_cat:
                continue
            text = re.sub(r"\s+", " ", row["Resume_str"]).strip()
            if len(text) < MIN_CHARS:
                skipped.append((row["ID"], cat, len(text)))
                continue
            by_cat[cat].append({"cv_id": str(row["ID"]).strip(), "category": cat, "text": text})

    rng = random.Random(SEED)
    out = []
    for cat in CATEGORIES:
        pool = sorted(by_cat[cat], key=lambda r: int(r["cv_id"]))
        out.extend(rng.sample(pool, PER_CATEGORY))

    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out)} CVs to {OUT}")
    for cv_id, cat, n in skipped:
        print(f"skipped (too short, {n} chars): {cv_id} {cat}")


if __name__ == "__main__":
    main()
