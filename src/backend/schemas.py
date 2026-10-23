"""Shared models (design 11.3.1) and API request/response bodies (11.3.5)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import DEFAULT_TOP_K, MAX_TOP_K


# ---- Shared models ----

class CVExtraction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)

    full_name: str
    email: str | None = None
    phone: str | None = None
    years_experience: float | None = Field(None, ge=0)
    skills: list[str]

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, values):
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    education: str | None = None
    summary: str | None = None


class SearchCriteria(BaseModel):
    skills: list[str] = Field(default_factory=list)
    min_years: float | None = Field(None, ge=0, allow_inf_nan=False)
    level: Literal["intern", "junior", "middle", "senior"] | None = None
    keywords: list[str] = Field(default_factory=list)


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
    query: str = Field(min_length=1, max_length=3000)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("Truy vấn không được để trống")
        return value.strip()

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
