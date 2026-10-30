"""Search, parallel ranking and durable human review via real LangGraph interrupts."""

import logging
import operator
import sqlite3
from contextlib import contextmanager
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from .. import config, db, services
from ..tracing import traced
from ..schemas import SearchCriteria
from ..scoring import score_cv as calculate_score

logger = logging.getLogger(__name__)


class SearchState(TypedDict, total=False):
    query: str
    top_k: int
    criteria: dict
    candidates: list[dict]
    results: Annotated[list[dict], operator.add]
    shortlist: list[dict]
    explanations: Annotated[list[dict], operator.add]
    approved_cv_ids: list[str]
    approval_status: Literal["pending", "approved", "rejected"]
    questions: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]


def node_error(node: str, exc: Exception, cv_id: str | None = None) -> dict:
    logger.warning("node=%s cv_id=%s error=%s", node, cv_id, type(exc).__name__)
    return {"errors": [{"node": node, "cv_id": cv_id, "code": type(exc).__name__}]}


def parse_query(state: SearchState) -> dict:
    try:
        return {"criteria": SearchCriteria.model_validate(
            services.parse_query(state["query"])).model_dump()}
    except Exception as exc:
        return {"criteria": SearchCriteria(keywords=[state["query"]]).model_dump(),
                **node_error("parse_query", exc)}


def retrieve(state: SearchState) -> dict:
    # Retrieval errors abort the search; the API marks its business row failed.
    return {"candidates": services.vector_search(state["query"], config.RETRIEVE_K)}


def score_cv(state: dict) -> dict:
    candidate = state["candidate"]
    try:
        cv = db.get_cv(candidate["cv_id"])
        if not cv or cv["status"] != "ready":
            raise LookupError("CV_NOT_READY")
        return {"results": [calculate_score(candidate["cv_id"], cv, state["criteria"],
                                             candidate["similarity"])]}
    except Exception as exc:
        # A stale vector must not produce a phantom candidate in the shortlist.
        return node_error("score_cv", exc, candidate["cv_id"])


def rank(state: SearchState) -> dict:
    by_id = {result["cv_id"]: result for result in state["results"]}
    return {"shortlist": sorted(by_id.values(), key=lambda r: (-r["score"], r["cv_id"]))[
        :state["top_k"]]}


def explain(state: dict) -> dict:
    result = state["result"]
    cv_id = result["cv_id"]
    try:
        cv = db.get_cv(cv_id)
        if not cv:
            raise LookupError("CV_NOT_FOUND")
        return {"explanations": [{"cv_id": cv_id, "reasons": services.explain_match(
            cv, state["criteria"], result)}]}
    except Exception as exc:
        return {"explanations": [{"cv_id": cv_id, "reasons": None}],
                **node_error("explain", exc, cv_id)}


def human_review(state: SearchState) -> dict:
    # This node is replayed on resume. No side effects before interrupt().
    payload = interrupt({"shortlist": state["shortlist"],
                         "explanations": state["explanations"]})
    allowed = {row["cv_id"] for row in state["shortlist"]}
    approved = [cv_id for cv_id in dict.fromkeys(payload["approved_cv_ids"]) if cv_id in allowed]
    return {"approved_cv_ids": approved,
            "approval_status": "approved" if approved else "rejected"}


def generate_questions(state: dict) -> dict:
    cv_id = state["cv_id"]
    try:
        cv = db.get_cv(cv_id)
        if not cv:
            raise LookupError("CV_NOT_FOUND")
        return {"questions": services.generate_questions(cv_id, cv, state["criteria"])}
    except Exception as exc:
        return node_error("generate_questions", exc, cv_id)


def build_search_graph(checkpointer):
    graph = StateGraph(SearchState)
    for node in (parse_query, retrieve, score_cv, rank, explain, human_review, generate_questions):
        graph.add_node(node.__name__, traced(node))
    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_conditional_edges("retrieve", lambda s: [
        Send("score_cv", {"candidate": c, "criteria": s["criteria"]})
        for c in s["candidates"]] or END, ["score_cv", END])
    graph.add_edge("score_cv", "rank")

    def route_explanations(state):
        if not state["shortlist"]:
            return END
        return [Send("explain", {"result": r, "criteria": state["criteria"]})
                for r in state["shortlist"] if r["score"] >= config.EXPLAIN_THRESHOLD] or "human_review"

    graph.add_conditional_edges("rank", route_explanations, ["explain", "human_review", END])
    graph.add_edge("explain", "human_review")
    graph.add_conditional_edges("human_review", lambda s: [
        Send("generate_questions", {"cv_id": cv_id, "criteria": s["criteria"]})
        for cv_id in s["approved_cv_ids"]] or END, ["generate_questions", END])
    graph.add_edge("generate_questions", END)
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def persistent_graph():
    # Independent connections avoid sharing a SQLite handle across HTTP worker threads.
    conn = sqlite3.connect(config.CHECKPOINT_DB_PATH, timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield build_search_graph(SqliteSaver(conn))
    finally:
        conn.close()


def graph_config(search_id: str) -> dict:
    return {"configurable": {"thread_id": search_id},
            "recursion_limit": config.RECURSION_LIMIT, "max_concurrency": config.MAX_CONCURRENCY}


def start_search(search_id: str, query: str, top_k: int) -> dict:
    with persistent_graph() as graph:
        result = graph.invoke({"query": query, "top_k": top_k, "criteria": {},
                               "candidates": [], "results": [], "shortlist": [],
                               "explanations": [], "approved_cv_ids": [],
                               "approval_status": "pending", "questions": [], "errors": []},
                              graph_config(search_id))
    return {"paused": bool(result.get("__interrupt__")), "criteria": result["criteria"],
            "shortlist": result["shortlist"], "explanations": result["explanations"], "errors": result.get("errors", [])}


def resume_search(search_id: str, approved_cv_ids: list[str]) -> list[dict]:
    cfg = graph_config(search_id)
    with persistent_graph() as graph:
        snapshot = graph.get_state(cfg)
        if not snapshot.values:
            raise LookupError("Không tìm thấy checkpoint cho lượt tìm kiếm.")
        allowed = {r["cv_id"] for r in snapshot.values["shortlist"]}
        approved = [i for i in dict.fromkeys(approved_cv_ids) if i in allowed]
        if snapshot.values.get("approval_status") != "pending":
            if set(approved) != set(snapshot.values["approved_cv_ids"]):
                raise ValueError("Quyết định duyệt khác với quyết định đã lưu.")
            # Recover if graph completion succeeded but the app.db write failed.
            result = graph.invoke(None, cfg) if snapshot.next else snapshot.values
        else:
            if "human_review" not in snapshot.next:
                raise ValueError("Graph không ở điểm chờ duyệt.")
            result = graph.invoke(Command(resume={"approved_cv_ids": approved}), cfg)
    order = {r["cv_id"]: i for i, r in enumerate(result["shortlist"])}
    return sorted(result["questions"], key=lambda q: order[q["cv_id"]])


def read_errors(search_id: str) -> list[dict]:
    """Recover historical errors using the same persistent thread, without resuming."""
    with persistent_graph() as graph:
        state = graph.get_state(graph_config(search_id))
    if not state.values:
        return [{"node": "checkpoint", "code": "HISTORICAL_ERRORS_UNAVAILABLE"}]
    return state.values.get("errors") or []
