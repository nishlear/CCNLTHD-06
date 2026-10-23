"""Parallel PDF ingest with per-CV validation and a bounded extraction cycle."""

import logging
import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .. import config, db, services
from ..tracing import traced
from ..schemas import CVExtraction
from ..text_cleaning import prepare_llm_input

logger = logging.getLogger(__name__)


class FileInput(TypedDict):
    cv_id: str
    file_name: str
    file_path: str
    raw_text: NotRequired[str]
    display_name: NotRequired[str]


class IngestOutcome(TypedDict):
    cv_id: str
    file_name: str
    status: Literal["ready", "failed"]
    error: str | None
    source: NotRequired[str | None]


class IngestState(TypedDict):
    files: list[FileInput]
    processed: Annotated[list[IngestOutcome], operator.add]


class ProcessCVState(TypedDict, total=False):
    cv_id: str
    file_name: str
    file_path: str
    raw_text: str
    clean_text: str
    display_name: str
    extracted: dict | None
    source: str | None
    missing_fields: list[str]
    attempts: int
    error: str | None
    processed: Annotated[list[IngestOutcome], operator.add]


def parse_pdf(state: ProcessCVState) -> dict:
    try:
        return {"raw_text": services.pdf_to_text(state["file_path"]), "error": None}
    except Exception as exc:
        logger.warning("node=parse_pdf cv_id=%s error=%s", state["cv_id"], type(exc).__name__)
        message = str(exc) if isinstance(exc, ValueError) else "Không thể đọc văn bản từ PDF."
        return {"error": message}


def preprocess(state: ProcessCVState) -> dict:
    """Non-LLM cleanup (Rule 2.1): regex contact redaction, whitespace, input cap."""
    clean = prepare_llm_input(state.get("raw_text") or "", config.LLM_INPUT_MAX_CHARS)
    if not clean:
        return {"clean_text": "", "error": "CV không có văn bản sau khi tiền xử lý."}
    return {"clean_text": clean, "error": None}


def extract(state: ProcessCVState) -> dict:
    attempts = state.get("attempts", 0) + 1
    try:
        data, source = services.extract_cv(state.get("clean_text") or "",
                                           state.get("missing_fields") or [])
        extracted = CVExtraction.model_validate(data)
        return {"extracted": extracted.model_dump(), "source": source,
                "attempts": attempts, "error": None}
    except Exception as exc:
        logger.warning("node=extract cv_id=%s attempt=%s error=%s",
                       state.get("cv_id"), attempts, type(exc).__name__)
        return {"extracted": None, "attempts": attempts,
                "error": "Không trích xuất được dữ liệu CV hợp lệ."}


def validate(state: ProcessCVState) -> dict:
    extracted = dict(state.get("extracted") or {})
    # Anonymized research corpora intentionally have no personal name. This is
    # an explicit display identifier supplied by the importer, not a guessed name.
    if state.get("extracted") is not None and not extracted.get("full_name") and state.get("display_name"):
        extracted["full_name"] = state["display_name"]
    missing = [key for key in ("full_name", "skills") if not extracted.get(key)]
    if not missing:
        return {"missing_fields": [], "error": None, "extracted": extracted}
    return {"missing_fields": missing,
            "error": f"Thiếu {', '.join(missing)} sau {state.get('attempts', 0)} lần thử."}


def outcome(state: ProcessCVState, status: str, error: str | None = None) -> dict:
    return {"processed": [{"cv_id": state.get("cv_id", ""), "file_name": state.get("file_name", ""),
                           "status": status, "error": error, "source": state.get("source")}]}


def save_and_index(state: ProcessCVState) -> dict:
    extracted = state.get("extracted")
    # Rule 4.1: never touch an empty or non-dict extraction result.
    if not isinstance(extracted, dict) or not extracted:
        logger.warning("node=save_and_index cv_id=%s error=EmptyExtraction", state.get("cv_id"))
        return {"error": "Kết quả trích xuất rỗng hoặc sai kiểu dữ liệu."}
    cv_id, file_name = state.get("cv_id", ""), state.get("file_name", "")
    raw_text = state.get("raw_text", "")
    try:
        # A failed row remains available even if vector indexing fails.
        db.save_cv(cv_id, file_name, "failed", raw_text=raw_text,
                   extracted=extracted, error="Đang lập chỉ mục.")
        services.index_cv(cv_id, extracted)
        # Rule 4.4: the ready checkpoint is committed to disk per CV.
        db.save_cv(cv_id, file_name, "ready", raw_text=raw_text, extracted=extracted)
        return outcome(state, "ready")
    except Exception as exc:
        logger.warning("node=save_and_index cv_id=%s error=%s", cv_id, type(exc).__name__)
        return {"error": "Không thể lưu hoặc lập chỉ mục CV."}


def mark_failed(state: ProcessCVState) -> dict:
    error = state.get("error") or "Không thể xử lý CV."
    db.save_cv(state.get("cv_id", ""), state.get("file_name", ""), "failed",
               raw_text=state.get("raw_text"), extracted=state.get("extracted"), error=error)
    return outcome(state, "failed", error)


def build_process_graph():
    graph = StateGraph(ProcessCVState)
    for node in (parse_pdf, preprocess, extract, validate, save_and_index, mark_failed):
        graph.add_node(node.__name__, traced(node))
    graph.add_conditional_edges(START, lambda s: "preprocess" if s.get("raw_text") else "parse_pdf")
    graph.add_conditional_edges("parse_pdf", lambda s: "mark_failed" if s.get("error") else "preprocess")
    graph.add_conditional_edges("preprocess", lambda s: "mark_failed" if s.get("error") else "extract")
    graph.add_edge("extract", "validate")

    def route_validation(state):
        if not state.get("missing_fields"):
            return "save_and_index"
        return "extract" if state.get("attempts", 0) < config.MAX_EXTRACT_ATTEMPTS else "mark_failed"

    graph.add_conditional_edges("validate", route_validation)
    graph.add_conditional_edges("save_and_index", lambda s: "mark_failed" if s.get("error") else END)
    graph.add_edge("mark_failed", END)
    return graph.compile()


def build_ingest_graph():
    graph = StateGraph(IngestState)
    graph.add_node("process_cv", build_process_graph())
    graph.add_conditional_edges(START, lambda s: [
        Send("process_cv", {**f, "attempts": 0, "missing_fields": [], "processed": []})
        for f in s["files"]] or END)
    graph.add_edge("process_cv", END)
    return graph.compile()


def run_ingest(files: list[FileInput]) -> list[IngestOutcome]:
    if not files:
        return []
    result = build_ingest_graph().invoke({"files": files, "processed": []},
                                        {"recursion_limit": config.RECURSION_LIMIT,
                                         "max_concurrency": config.MAX_CONCURRENCY})
    by_id = {row["cv_id"]: row for row in result["processed"]}
    return [by_id[file["cv_id"]] for file in files]
