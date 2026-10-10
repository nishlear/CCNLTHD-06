"""Ingest graph (design 11.2.1). PLACEHOLDER: replace run_ingest with the real LangGraph."""

from typing import Literal, TypedDict

from .. import db


class FileInput(TypedDict):
    cv_id: str
    file_name: str
    file_path: str


class IngestOutcome(TypedDict):
    cv_id: str
    file_name: str
    status: Literal["ready", "failed"]
    error: str | None


def run_ingest(files: list[FileInput]) -> list[IngestOutcome]:
    for f in files:
        db.save_cv(f["cv_id"], f["file_name"], "ready",
                   extracted={"full_name": "change me", "skills": ["change me"]})
    return [{"cv_id": f["cv_id"], "file_name": f["file_name"], "status": "ready", "error": None}
            for f in files]
