"""Local corpus console: allowlisted operations, bounded batches and saved jobs."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .. import corpus, services

router = APIRouter(prefix="/corpus", tags=["corpus"])


class CorpusJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["prepare", "verify", "ingest", "benchmark"]
    corpus_id: str | None = Field(None, max_length=100)
    limit: int = Field(20, ge=1, le=50)
    batch_size: int = Field(5, ge=1, le=20)
    mode: Literal["next", "range"] = "next"
    offset: int = Field(0, ge=0, le=100000)
    sample_count: int = Field(100000, ge=1, le=100000)
    seed: int = Field(20261023, ge=0, le=2147483647)
    repeats: int = Field(20, ge=1, le=100)


@router.get("")
def overview():
    return corpus.overview()


@router.get("/{corpus_id}/preview")
def preview(corpus_id: str, offset: int = Query(0, ge=0, le=100000), limit: int = Query(10, ge=1, le=20)):
    try:
        return corpus.preview(corpus_id, offset, limit)
    except (ValueError, OSError):
        raise HTTPException(404, "Không tìm thấy tập hồ sơ hợp lệ.") from None


@router.post("/jobs", status_code=202)
def start_job(body: CorpusJobRequest):
    try:
        return corpus.start_job(body.model_dump())
    except corpus.JobConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except services.ConfigurationError as exc:
        raise HTTPException(503, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/jobs/{job_id}")
def job(job_id: str):
    result = corpus.get_job(job_id)
    if result is None:
        raise HTTPException(404, "Không tìm thấy tác vụ.")
    return result


@router.post("/jobs/{job_id}/stop")
def stop(job_id: str):
    try:
        return corpus.stop_job(job_id)
    except KeyError:
        raise HTTPException(404, "Không tìm thấy tác vụ.") from None
    except corpus.JobConflict as exc:
        raise HTTPException(409, str(exc)) from None


@router.get("/jobs/{job_id}/report")
def report(job_id: str):
    result = job(job_id)
    return JSONResponse(result, headers={"Content-Disposition": f'attachment; filename="corpus-{result["id"]}.json"'})
