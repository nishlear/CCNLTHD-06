"""Read persisted application/checkpoint/vector evidence without calling a model."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.backend import config, db, services
from src.backend.graphs.search import graph_config, persistent_graph
from src.backend.text_cleaning import redact_contacts


def capture(search_id=None):
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Read-only persisted state; no model calls", "models": services.model_settings()}
    with db.connect() as conn:
        report["cv_status_counts"] = dict(conn.execute("SELECT status,count(*) FROM cv GROUP BY status"))
        report["extraction_cache_count"] = conn.execute("SELECT count(*) FROM extraction_cache").fetchone()[0]
        ready_ids = {row[0] for row in conn.execute("SELECT id FROM cv WHERE status='ready'")}
    store = services.collection()
    result = store.get(include=["embeddings", "metadatas"])
    raw_vectors = result.get("embeddings")
    vectors = list(raw_vectors) if raw_vectors is not None else []
    norms = [math.sqrt(sum(float(value)**2 for value in vector)) for vector in vectors]
    ids = result.get("ids") or []
    report["chroma"] = {"count": store.count(), "metadata": store.metadata,
        "dimensions": sorted({len(vector) for vector in vectors}),
        "all_finite": all(math.isfinite(float(value)) for vector in vectors for value in vector),
        "norm_min": min(norms) if norms else None, "norm_max": max(norms) if norms else None,
        "doc_formats": dict(Counter((row or {}).get("doc_format", "missing") for row in result.get("metadatas") or [])),
        "ready_without_vector": sorted(ready_ids-set(ids))}
    paths = [Path("pyproject.toml"), Path("uv.lock"), *Path("src/backend").rglob("*.py")]
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if search_id:
        search = db.get_search(search_id)
        if not search:
            raise ValueError("Search ID does not exist")
        shortlist = db.get_shortlist(search_id)
        questions = db.get_questions(search_id)
        profiles = []
        for row in shortlist:
            cv = db.get_cv(row.get("cv_id", "")) or {}
            text, _ = redact_contacts(cv.get("raw_text") or "")
            excerpts = [match.group().strip() for match in re.finditer(
                r"[^.\n]{0,50}\b\d+\+?\s+years?[^.\n]{0,90}", text, re.I)][:5]
            profiles.append({key: cv.get(key) for key in ("id", "full_name", "role", "years_experience", "skills")}
                            | {"source_excerpts": excerpts})
        with persistent_graph() as graph:
            cfg = graph_config(search_id)
            state = graph.get_state(cfg)
            history = [{"created_at": item.created_at, "next": list(item.next),
                        "approval_status": item.values.get("approval_status"),
                        "question_count": len(item.values.get("questions") or []),
                        "errors": item.values.get("errors") or []}
                       for item in graph.get_state_history(cfg)]
        report["search"] = {**search, "shortlist": shortlist, "profiles": profiles,
                            "questions": questions, "question_categories": dict(Counter(q.get("category") for q in questions))}
        report["checkpoint"] = {"created_at": state.created_at, "next": list(state.next),
            "approval_status": state.values.get("approval_status"),
            "approved_cv_ids": state.values.get("approved_cv_ids") or [],
            "question_count": len(state.values.get("questions") or []), "errors": state.values.get("errors") or [],
            "history": history}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-id")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expect", choices=("waiting", "completed"))
    args = parser.parse_args()
    report = capture(args.search_id)
    checks = {"vectors_768d": report["chroma"]["dimensions"] == [768],
              "vectors_finite": report["chroma"]["all_finite"],
              "ready_vectors_present": not report["chroma"]["ready_without_vector"]}
    if args.expect:
        state, search = report["checkpoint"], report["search"]
        checks["no_graph_errors"] = not state["errors"]
        if args.expect == "waiting":
            checks.update(waiting=search.get("status") == "awaiting_review" and "human_review" in state["next"],
                          no_questions=state["question_count"] == len(search["questions"]) == 0,
                          pending=state["approval_status"] == "pending")
        else:
            checks.update(completed=search.get("status") == "completed" and not state["next"],
                          one_approved=len(state["approved_cv_ids"]) == 1,
                          six_questions=state["question_count"] == len(search["questions"]) == 6,
                          two_per_category=search["question_categories"] == {"technical":2,"experience":2,"behavioral":2})
    report["checks"] = checks
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "checks": checks}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
