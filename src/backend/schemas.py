"""Shared models (design 11.3.1) and API request/response bodies (11.3.5)."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import DEFAULT_TOP_K, MAX_TOP_K


# ---- Shared models ----

def _clean_skill_list(values) -> list[str]:
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(v).strip() for v in values if v is not None and str(v).strip()))


class CVExtract(BaseModel):
    """Unified LLM output contract (Rules 1.3, 2.3): identical for Gemini and Ollama.

    Short keys keep output tokens low. Validators are lenient because the local
    fallback model may emit strings like "5+" or a comma-joined skill list.
    """
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False, extra="ignore")

    name: str = ""
    role: str = ""
    skills: list[str] = Field(default_factory=list)
    exp_yrs: float | None = Field(None, ge=0, le=70)
    edu: str = ""
    summary: str = ""

    @field_validator("name", "role", "edu", "summary", mode="before")
    @classmethod
    def coerce_text(cls, value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value if v is not None)
        return str(value)

    @field_validator("skills", mode="before")
    @classmethod
    def coerce_skills(cls, values):
        return _clean_skill_list(values)

    @field_validator("exp_yrs", mode="before")
    @classmethod
    def coerce_years(cls, value):
        if value is None or isinstance(value, bool):
            return None
        if not isinstance(value, (int, float)):
            match = re.search(r"\d+(?:[.,]\d+)?", str(value))
            value = float(match.group().replace(",", ".")) if match else None
        return value if value is not None and 0 <= value <= 70 else None

    def to_record(self) -> dict:
        """Map the compact contract to the storage/record keys used downstream."""
        return {"full_name": self.name, "role": self.role or None, "skills": self.skills,
                "years_experience": self.exp_yrs, "education": self.edu or None,
                "summary": self.summary or None}


class CVExtraction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, allow_inf_nan=False)

    full_name: str
    email: str | None = None
    phone: str | None = None
    role: str | None = None
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
    role: str | None = None
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


class SearchOutcomeFields(BaseModel):
    outcome: Literal["pending", "succeeded", "completed_with_errors", "failed"] = "pending"
    errors: list[dict] = Field(default_factory=list)
    approved_count: int = 0
    question_count: int = 0
    expected_question_count: int = 0
    incomplete_cv_ids: list[str] = Field(default_factory=list)
    ranking_mode: Literal["soft_priority"] = "soft_priority"
    hard_filters_applied: bool = False


class SearchResponse(SearchOutcomeFields):
    id: str
    status: SearchStatus
    criteria: SearchCriteria | None
    shortlist: list[ShortlistItem]


class ReviewRequest(BaseModel):
    approved_cv_ids: list[str]


class ReviewResponse(SearchOutcomeFields):
    id: str
    status: SearchStatus
    questions: list[Question]


class SearchListItem(SearchOutcomeFields):
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
