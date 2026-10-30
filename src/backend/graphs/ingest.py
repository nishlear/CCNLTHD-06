"""Two-phase ingest: extract/cache every CV, then batch embed/upsert."""

import hashlib
import json
import logging
import operator
import threading
import os
import time
from contextlib import contextmanager
from typing import Annotated, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .. import config, db, services
from ..tracing import traced
from ..schemas import CVExtraction
from ..text_cleaning import prepare_llm_input, redact_contacts

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
    already_ready: NotRequired[bool]


class IngestState(TypedDict):
    files: list[FileInput]
    prepared: Annotated[list[dict], operator.add]
    processed: Annotated[list[IngestOutcome], operator.add]


class ProcessCVState(TypedDict, total=False):
    cv_id: str
    file_name: str
    file_path: str
    raw_text: str
    clean_text: str
    display_name: str
    fingerprint: str
    cache_hit: bool
    prepared: Annotated[list[dict], operator.add]
    extracted: dict | None
    source: str | None
    missing_fields: list[str]
    attempts: int
    error: str | None
    processed: Annotated[list[IngestOutcome], operator.add]


def parse_pdf(state: ProcessCVState) -> dict:
    try:
        return {"raw_text": services.pdf_to_text(state.get("file_path", "")), "error": None}
    except Exception as exc:
        logger.warning("node=parse_pdf cv_id=%s error=%s", state.get("cv_id", ""), type(exc).__name__)
        message = str(exc) if isinstance(exc, ValueError) else "Không thể đọc văn bản từ PDF."
        return {"error": message}


def preprocess(state: ProcessCVState) -> dict:
    """Non-LLM cleanup (Rule 2.1): regex contact redaction, whitespace, input cap."""
    raw, _ = redact_contacts(state.get("raw_text") or "")
    clean = prepare_llm_input(raw, config.LLM_INPUT_MAX_CHARS)
    if not clean:
        return {"clean_text": "", "error": "Empty CV after preprocessing."}
    fingerprint = hashlib.sha256(json.dumps(
        ["extract-short-v1", clean, state.get("display_name", "")], ensure_ascii=False).encode()).hexdigest()
    cached = db.get_extraction(state.get("cv_id", ""), fingerprint)
    if cached:
        try:
            extracted = CVExtraction.model_validate(cached.get("extracted")).model_dump()
            if not extracted.get("full_name") or not extracted.get("skills"):
                raise ValueError("Invalid cache")
            return {"raw_text": raw, "clean_text": clean, "fingerprint": fingerprint,
                    "cache_hit": True, "extracted": extracted, "source": cached.get("source"), "error": None}
        except (TypeError, ValueError):
            pass
    return {"raw_text": raw, "clean_text": clean, "fingerprint": fingerprint,
            "cache_hit": False, "error": None}


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
    data = state.get("extracted")
    if not isinstance(data, dict) or not data:
        return {"missing_fields": ["extracted"], "error": state.get("error") or "Invalid extraction."}
    extracted = dict(data)
    # Anonymized research corpora intentionally have no personal name. This is
    # an explicit display identifier supplied by the importer, not a guessed name.
    if state.get("extracted") is not None and not extracted.get("full_name") and state.get("display_name"):
        extracted["full_name"] = state.get("display_name", "")
    missing = [key for key in ("full_name", "skills") if not extracted.get(key)]
    if not missing:
        return {"missing_fields": [], "error": None, "extracted": extracted}
    return {"missing_fields": missing,
            "error": f"Thiếu {', '.join(missing)} sau {state.get('attempts', 0)} lần thử."}


def outcome(state: ProcessCVState, status: str, error: str | None = None) -> dict:
    return {"processed": [{"cv_id": state.get("cv_id", ""), "file_name": state.get("file_name", ""),
                           "status": status, "error": error, "source": state.get("source")}]}


def save_and_index(state: ProcessCVState) -> dict:
    """Single-record entry point using the same defensive batch indexing path."""
    extracted = state.get("extracted")
    if not isinstance(extracted, dict) or not extracted:
        error = "Empty or invalid extraction."
        return {**mark_failed({**state, "error": error}), "error": error}
    return save_and_index_batch({"prepared": [state]})


def mark_failed(state: ProcessCVState) -> dict:
    error = state.get("error") or "Không thể xử lý CV."
    cv_id = state.get("cv_id", "")
    if not isinstance(cv_id, str):
        cv_id = ""
    logger.warning("node=mark_failed cv_id=%s status=failed", cv_id)
    try:
        data = state.get("extracted")
        # Invalid field types must not prevent persisting the failed status.
        try:
            data = CVExtraction.model_validate(data).model_dump() if isinstance(data, dict) else None
        except (TypeError, ValueError):
            data = None
        if cv_id:
            db.save_cv(cv_id, str(state.get("file_name") or ""), "failed",
                       raw_text=state.get("raw_text") if isinstance(state.get("raw_text"), str) else None,
                       extracted=data, error=error)
    except Exception as exc:
        logger.error("node=mark_failed cv_id=%s error=%s", state.get("cv_id"), type(exc).__name__)
    return outcome({**state, "cv_id": cv_id}, "failed", error)


def cache_extraction(state: ProcessCVState) -> dict:
    db.save_extraction(state.get("cv_id", ""), state.get("fingerprint", ""),
                       state.get("extracted") or {}, state.get("source"))
    logger.info("phase=extract_cache cv_id=%s cache_hit=%s source=%s", state.get("cv_id"),
                state.get("cache_hit", False), state.get("source"))
    return {"prepared": [{key: state.get(key) for key in
                          ("cv_id", "file_name", "raw_text", "extracted", "source")} ]}


def build_process_graph():
    """Phase 1 only. No embedding or vector-store calls in this graph."""
    graph = StateGraph(ProcessCVState)
    for node in (parse_pdf, preprocess, extract, validate, cache_extraction, mark_failed):
        graph.add_node(node.__name__, traced(node))
    graph.add_conditional_edges(START, lambda s: "preprocess" if s.get("raw_text") is not None else "parse_pdf")
    graph.add_conditional_edges("parse_pdf", lambda s: "mark_failed" if s.get("error") else "preprocess")
    graph.add_conditional_edges("preprocess", lambda s: "mark_failed" if s.get("error") else
                                ("validate" if s.get("cache_hit") else "extract"))
    graph.add_conditional_edges("extract", lambda s: "mark_failed" if s.get("error") else "validate")

    # One cloud attempt, then one local attempt only on quota/timeout errors.
    # A malformed/incomplete result fails this record rather than spending a
    # second cloud request or cycling back to Gemini after local failover.
    graph.add_conditional_edges("validate", lambda s: "mark_failed" if s.get("missing_fields") else "cache_extraction")
    graph.add_edge("cache_extraction", END)
    graph.add_edge("mark_failed", END)
    return graph.compile()


def save_and_index_batch(state: IngestState) -> dict:
    """Phase 2. Every extraction/cache write has finished before this node runs."""
    processed = []
    prepared = []
    for raw_item in state.get("prepared") or []:
        item = raw_item if isinstance(raw_item, dict) else {}
        try:
            if not isinstance(raw_item, dict):
                raise ValueError("Record is not a JSON object")
            if not isinstance(item.get("cv_id"), str) or not item.get("cv_id"):
                raise ValueError("Missing cv_id")
            data = item.get("extracted")
            if not isinstance(data, dict) or not data:
                raise ValueError("Empty or invalid extraction")
            validated = CVExtraction.model_validate(data).model_dump()
            if not validated.get("full_name") or not validated.get("skills"):
                raise ValueError("Incomplete extraction")
            item = {**item, "extracted": validated}
            # A crash during embedding leaves an explicit row plus the reusable cache.
            db.save_cv(item.get("cv_id", ""), item.get("file_name", ""), "failed",
                       raw_text=item.get("raw_text"), extracted=validated, error="Pending vector indexing.")
            prepared.append(item)
        except Exception as exc:
            processed.extend(mark_failed({**item, "error": f"Invalid record: {type(exc).__name__}"}).get("processed", []))

    for start in range(0, len(prepared), config.EMBED_BATCH_SIZE):
        batch = prepared[start:start + config.EMBED_BATCH_SIZE]
        logger.info("phase=embed_upsert batch_size=%s offset=%s", len(batch), start)
        try:
            vectors = services.embed_documents_with_backoff([
                services.embedding_text(item.get("extracted") or {}) for item in batch])
        except Exception as exc:
            for item in batch:
                processed.extend(mark_failed({**item, "error": f"Embedding failed: {type(exc).__name__}"}).get("processed", []))
            continue
        try:
            services.upsert_vectors([item.get("cv_id", "") for item in batch], vectors)
        except Exception as exc:
            logger.warning("node=save_and_index bulk_upsert_error=%s", type(exc).__name__)
            # Reuse vectors: isolate an individual Chroma rejection without
            # making more API requests or failing the other CVs in the batch.
            for item, vector in zip(batch, vectors):
                try:
                    services.upsert_vectors([item.get("cv_id", "")], [vector])
                    db.save_cv(item.get("cv_id", ""), item.get("file_name", ""), "ready",
                               raw_text=item.get("raw_text"), extracted=item.get("extracted"))
                    processed.extend(outcome(item, "ready").get("processed", []))
                except Exception as error:
                    processed.extend(mark_failed({**item, "error": f"Index failed: {type(error).__name__}"}).get("processed", []))
            continue
        for item in batch:
            try:
                # Commit ready immediately per record, rather than per batch.
                db.save_cv(item.get("cv_id", ""), item.get("file_name", ""), "ready",
                           raw_text=item.get("raw_text"), extracted=item.get("extracted"))
                processed.extend(outcome(item, "ready").get("processed", []))
            except Exception as exc:
                processed.extend(mark_failed({**item, "error": f"Checkpoint failed: {type(exc).__name__}"}).get("processed", []))
    return {"processed": processed}


def build_ingest_graph():
    phase_one = build_process_graph()

    def process_cv(state):
        try:
            result = phase_one.invoke(state, {"recursion_limit": config.RECURSION_LIMIT})
            return {"prepared": result.get("prepared", []), "processed": result.get("processed", [])}
        except Exception as exc:
            # Covers cache/SQLite failures as well as ordinary provider errors.
            return mark_failed({**state, "error": f"Extraction failed: {type(exc).__name__}"})

    graph = StateGraph(IngestState)
    graph.add_node("process_cv", process_cv)
    graph.add_node("save_and_index", traced(save_and_index_batch))
    graph.add_conditional_edges(START, lambda s: [
        Send("process_cv", {**f, "attempts": 0, "missing_fields": [], "processed": [], "prepared": []})
        for f in s.get("files", [])] or END)
    # LangGraph's superstep barrier waits for all process_cv branches.
    graph.add_edge("process_cv", "save_and_index")
    graph.add_edge("save_and_index", END)
    return graph.compile()


_ingest_lock = threading.Lock()


@contextmanager
def pipeline_file_lock():
    """OS lock serializes phases across processes sharing app.db; crash releases it."""
    path = db.APP_DB_PATH.with_suffix(".ingest.lock")
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (13, 36):
                        raise
                    time.sleep(.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_ingest(files: list[FileInput]) -> list[IngestOutcome]:
    if not files:
        return []
    # Multiple uploads/processes sharing app.db cannot interleave ingest phases.
    with _ingest_lock, pipeline_file_lock():
        pending, by_position, pending_positions = [], {}, {}
        for position, raw_file in enumerate(files):
            file = raw_file if isinstance(raw_file, dict) else {}
            cv_id = file.get("cv_id", "")
            try:
                if not isinstance(raw_file, dict):
                    raise ValueError("Record is not a JSON object")
                if not isinstance(cv_id, str) or not cv_id:
                    raise ValueError("Missing cv_id")
                existing = db.get_cv(cv_id) or {}
                # Dataset runner additionally checks matching source text. For
                # direct callers, compare text here too before reusing a record.
                if (existing.get("status") == "ready" and
                    (file.get("raw_text") is None or existing.get("raw_text") == redact_contacts(file.get("raw_text") or "")[0]) and
                    services.has_vector(cv_id)):
                    by_position[position] = {"cv_id": cv_id, "file_name": file.get("file_name", ""),
                                             "status": "ready", "error": None, "already_ready": True}
                else:
                    pending.append(file)
                    pending_positions.setdefault(cv_id, []).append(position)
            except Exception as exc:
                by_position[position] = mark_failed({**file, "error": f"Record failed: {type(exc).__name__}"}).get("processed", [])[0]
        if pending:
            result = build_ingest_graph().invoke({"files": pending, "processed": [], "prepared": []},
                                                {"recursion_limit": config.RECURSION_LIMIT,
                                                 "max_concurrency": config.MAX_CONCURRENCY})
            for row in result.get("processed", []):
                for position in pending_positions.get(row.get("cv_id", ""), []):
                    by_position[position] = row
        return [by_position.get(position) or mark_failed({
                    **(file if isinstance(file, dict) else {}), "error": "Missing ingest outcome."
                }).get("processed", [])[0] for position, file in enumerate(files)]
