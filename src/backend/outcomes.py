"""Separate workflow termination from usable output; never infer hiring eligibility."""
from collections import Counter

from . import db
from .graphs.search import read_errors


def outcome_fields(search_id: str, row: dict | None = None) -> dict:
    row = row or db.get_search(search_id) or {}
    status = row.get("status")
    shortlist, questions = db.get_shortlist(search_id), db.get_questions(search_id)
    approved = [r.get("cv_id") for r in shortlist if r.get("approved")]
    errors = row.get("errors")
    if errors is None:
        try:
            errors = read_errors(search_id) if status == "completed" else []
        except Exception:
            errors = [{"node": "checkpoint", "code": "CHECKPOINT_UNAVAILABLE"}]
    counts = Counter((q.get("cv_id"), q.get("category")) for q in questions)
    incomplete = [cv_id for cv_id in approved if any(counts[(cv_id, category)] != 2
                  for category in ("technical", "experience", "behavioral"))]
    outcome = ("failed" if status == "failed" else "pending" if status != "completed" else
               "completed_with_errors" if errors or incomplete else "succeeded")
    return {"outcome": outcome, "errors": errors, "approved_count": len(approved),
            "question_count": len(questions), "expected_question_count": 6 * len(approved),
            "incomplete_cv_ids": incomplete, "ranking_mode": "soft_priority", "hard_filters_applied": False}
