"""Shared models (design 11.3.1) and API request/response bodies (11.3.5)."""

from typing import Literal

from pydantic import BaseModel, Field

from .config import DEFAULT_TOP_K, MAX_TOP_K


# ---- Shared models ----

class CVExtraction(BaseModel):
    full_name: str
    email: str | None = None
    phone: str | None = None
    years_experience: float | None = None
    skills: list[str]
    education: str | None = None
    summary: str | None = None


class SearchCriteria(BaseModel):
    skills: list[str] = []
    min_years: float | None = None
    level: Literal["intern", "junior", "middle", "senior"] | None = None
    keywords: list[str] = []


class Candidate(BaseModel):
    cv_id: str
    similarity: float


class ScoreResult(BaseModel):
    cv_id: str
    score: float
    matched_skills: list[str]
    missing_skills: list[str]


class Explanation(BaseModel):
    cv_id: str
    reasons: str | None


class Question(BaseModel):
    cv_id: str
    category: Literal["technical", "experience", "behavioral"]
    question: str


# ---- API bodies ----

CVStatus = Literal["ready", "failed"]
SearchStatus = Literal["running", "awaiting_review", "completed", "failed"]


class CVSummary(BaseModel):
    id: str
    file_name: str
    status: CVStatus
    error: str | None


class CVListItem(CVSummary):
    full_name: str | None
    skills: list[str]
    years_experience: float | None
    created_at: str


class CVDetail(CVListItem):
    email: str | None
    phone: str | None
    education: str | None
    summary: str | None


class UploadResponse(BaseModel):
    cvs: list[CVSummary]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class ShortlistItem(BaseModel):
    cv_id: str
    full_name: str | None
    score: float
    rank: int
    matched_skills: list[str]
    missing_skills: list[str]
    reasons: str | None
    approved: bool | None = None


class SearchResponse(BaseModel):
    id: str
    status: SearchStatus
    criteria: SearchCriteria | None
    shortlist: list[ShortlistItem]


class ReviewRequest(BaseModel):
    approved_cv_ids: list[str]


class ReviewResponse(BaseModel):
    id: str
    status: SearchStatus
    questions: list[Question]


class SearchListItem(BaseModel):
    id: str
    query: str
    top_k: int
    status: SearchStatus
    created_at: str


class SearchDetail(SearchResponse):
    query: str
    top_k: int
    created_at: str
    approved_cv_ids: list[str]
    questions: list[Question]
