import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .. import config, db, services
from ..graphs.ingest import FileInput, run_ingest
from ..schemas import CVDetail, CVListItem, CVStatus, UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cvs", tags=["cvs"])


def require_models():
    try:
        services.require_gemini()
    except services.ConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("", response_model=UploadResponse)
def upload_cvs(files: list[UploadFile] = File(...)):
    require_models()
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(400, f"Mỗi lần tải tối đa {config.MAX_UPLOAD_FILES} CV.")
    inputs: list[FileInput] = []
    try:
        for f in files:
            name = Path(f.filename or "cv.pdf").name
            if not name.lower().endswith(".pdf"):
                raise HTTPException(400, f"{name} không phải file PDF")
            content = f.file.read(config.MAX_UPLOAD_BYTES + 1)
            if len(content) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} vượt quá 10 MB.")
            if not content:
                raise HTTPException(400, f"{name} là file rỗng.")
            cv_id = uuid.uuid4().hex
            path = config.UPLOAD_DIR / f"{cv_id}.pdf"
            inputs.append({"cv_id": cv_id, "file_name": name, "file_path": str(path)})
            path.write_bytes(content)
    except Exception:
        # Rejecting a later file must not leave earlier staged uploads behind.
        for item in inputs:
            Path(item["file_path"]).unlink(missing_ok=True)
        raise

    try:
        outcomes = run_ingest(inputs)
    except Exception as exc:
        logger.error("operation=ingest error=%s", type(exc).__name__)
        for item in inputs:
            if not db.get_cv(item["cv_id"]):
                Path(item["file_path"]).unlink(missing_ok=True)
        raise HTTPException(500, "Không thể hoàn tất xử lý lô CV.") from exc
    return {"cvs": [{"id": o["cv_id"], "file_name": o["file_name"], "status": o["status"],
                     "error": o["error"]} for o in outcomes]}


@router.get("", response_model=list[CVListItem])
def list_cvs(status: CVStatus | None = None, limit: int = Query(100, ge=1, le=200),
             offset: int = Query(0, ge=0)):
    return db.list_cvs(status, limit=limit, offset=offset)


@router.get("/{cv_id}", response_model=CVDetail)
def get_cv(cv_id: str):
    cv = db.get_cv(cv_id)
    if not cv:
        raise HTTPException(404, "không tìm thấy CV")
    return cv


@router.delete("/{cv_id}", status_code=204)
def delete_cv(cv_id: str):
    if not db.get_cv(cv_id):
        raise HTTPException(404, "không tìm thấy CV")
    try:
        services.delete_vector(cv_id)
    except Exception as exc:
        logger.warning("operation=delete_vector cv_id=%s error=%s", cv_id, type(exc).__name__)
        raise HTTPException(503, "Chưa thể xóa CV khỏi chỉ mục. Hãy thử lại.") from exc
    (config.UPLOAD_DIR / f"{cv_id}.pdf").unlink(missing_ok=True)
    db.delete_cv(cv_id)
