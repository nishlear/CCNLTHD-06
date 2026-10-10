"""Search graph (design 11.2.2). PLACEHOLDER: replace both functions with the real LangGraph."""


def start_search(search_id: str, query: str, top_k: int) -> dict:
    """Run until interrupt(). paused=False means no candidates (graph ended)."""
    return {
        "paused": True,
        "criteria": {"skills": [], "min_years": None, "level": None, "keywords": ["change me"]},
        "shortlist": [{"cv_id": "change-me", "score": 0.5,
                       "matched_skills": [], "missing_skills": []}],
        "explanations": [{"cv_id": "change-me", "reasons": "change me"}],
    }


def resume_search(search_id: str, approved_cv_ids: list[str]) -> list[dict]:
    """Resume with Command(resume=...) and return the generated questions."""
    return [{"cv_id": i, "category": "technical", "question": "change me"}
            for i in approved_cv_ids]
