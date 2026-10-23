"""Deterministic ranking from report sections 11.3.2 and 12.1.5."""


def score_cv(cv_id: str, cv: dict, criteria: dict, similarity: float) -> dict:
    requested = {s.strip().casefold() for s in criteria.get("skills", []) if s.strip()}
    found = {s.strip().casefold() for s in cv.get("skills", []) if s.strip()}
    matched = sorted(requested & found)
    missing = sorted(requested - found)
    skill_fit = len(matched) / len(requested) if requested else 1.0
    minimum = criteria.get("min_years")
    experience_fit = (1.0 if minimum is None or minimum <= 0 else
                      min(max(cv.get("years_experience") or 0, 0) / minimum, 1.0))
    similarity = min(max(similarity, 0.0), 1.0)
    return {"cv_id": cv_id,
            "score": round(0.5 * skill_fit + 0.3 * similarity + 0.2 * experience_fit, 6),
            "matched_skills": matched, "missing_skills": missing}
