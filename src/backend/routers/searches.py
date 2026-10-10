import uuid

from fastapi import APIRouter, HTTPException

from .. import db
from ..graphs.search import resume_search, start_search
from ..schemas import (ReviewRequest, ReviewResponse, SearchDetail, SearchListItem,
                       SearchRequest, SearchResponse)

router = APIRouter(prefix="/searches", tags=["searches"])


def _response(search_id: str) -> dict:
    s = db.get_search(search_id)
    return {"id": s["id"], "status": s["status"], "criteria": s["criteria"],
            "shortlist": db.get_shortlist(search_id)}


@router.post("", response_model=SearchResponse)
def create_search(body: SearchRequest):
    search_id = uuid.uuid4().hex
    db.create_search(search_id, body.query, body.top_k)
    try:
        out = start_search(search_id, body.query, body.top_k)
    except Exception as e:
        db.update_search(search_id, status="failed")
        raise HTTPException(500, f"tìm kiếm thất bại: {e}") from e

    reasons = {e["cv_id"]: e["reasons"] for e in out["explanations"]}
    db.save_shortlist(search_id, [
        {**r, "rank": i + 1, "reasons": reasons.get(r["cv_id"])}
        for i, r in enumerate(out["shortlist"])
    ])
    db.update_search(search_id, status="awaiting_review" if out["paused"] else "completed",
                     criteria=out["criteria"])
    return _response(search_id)


@router.post("/{search_id}/review", response_model=ReviewResponse)
def review_search(search_id: str, body: ReviewRequest):
    s = db.get_search(search_id)
    if not s:
        raise HTTPException(404, "không tìm thấy lượt tìm kiếm")
    if s["status"] != "awaiting_review":
        raise HTTPException(409, f"lượt tìm kiếm đang ở trạng thái {s['status']}")

    shortlist_ids = {r["cv_id"] for r in db.get_shortlist(search_id)}
    approved = [i for i in dict.fromkeys(body.approved_cv_ids) if i in shortlist_ids]
    try:
        questions = resume_search(search_id, approved)
    except Exception as e:
        raise HTTPException(500, f"tiếp tục tìm kiếm thất bại: {e}") from e

    db.save_review(search_id, approved, questions)
    db.update_search(search_id, status="completed")
    return {"id": search_id, "status": "completed", "questions": questions}


@router.get("", response_model=list[SearchListItem])
def list_searches():
    return db.list_searches()


@router.get("/{search_id}", response_model=SearchDetail)
def get_search(search_id: str):
    s = db.get_search(search_id)
    if not s:
        raise HTTPException(404, "không tìm thấy lượt tìm kiếm")
    shortlist = db.get_shortlist(search_id)
    return {**s, "shortlist": shortlist,
            "approved_cv_ids": [r["cv_id"] for r in shortlist if r["approved"]],
            "questions": db.get_questions(search_id)}
