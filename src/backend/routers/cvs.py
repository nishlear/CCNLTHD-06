import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import db
from ..config import UPLOAD_DIR
from ..graphs.ingest import FileInput, run_ingest
from ..schemas import CVDetail, CVListItem, CVStatus, UploadResponse

router = APIRouter(prefix="/cvs", tags=["cvs"])


@router.post("", response_model=UploadResponse)
def upload_cvs(files: list[UploadFile] = File(...)):
    inputs: list[FileInput] = []
    for f in files:
        name = Path(f.filename or "cv.pdf").name
        if not name.lower().endswith(".pdf"):
            raise HTTPException(400, f"{name} không phải file PDF")
        cv_id = uuid.uuid4().hex
        path = UPLOAD_DIR / f"{cv_id}.pdf"
        path.write_bytes(f.file.read())
        inputs.append({"cv_id": cv_id, "file_name": name, "file_path": str(path)})

    outcomes = run_ingest(inputs)
    return {"cvs": [{"id": o["cv_id"], "file_name": o["file_name"], "status": o["status"],
                     "error": o["error"]} for o in outcomes]}


@router.get("", response_model=list[CVListItem])
def list_cvs(status: CVStatus | None = None):
    return db.list_cvs(status)


@router.get("/{cv_id}", response_model=CVDetail)
def get_cv(cv_id: str):
    cv = db.get_cv(cv_id)
    if not cv:
        raise HTTPException(404, "không tìm thấy CV")
    return cv


@router.delete("/{cv_id}", status_code=204)
def delete_cv(cv_id: str):
    if not db.delete_cv(cv_id):
        raise HTTPException(404, "không tìm thấy CV")
    (UPLOAD_DIR / f"{cv_id}.pdf").unlink(missing_ok=True)
    # TODO: also remove from Chroma once the real index exists
