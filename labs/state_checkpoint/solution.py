"""Deterministic teaching lab: real LangGraph/SQLite, no LLM or cloud requests."""
import argparse
import json
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

class State(TypedDict, total=False):
    items: list[str]
    collected: Annotated[list[str], operator.add]
    approved: bool
    output: list[str]

def collect(state):
    return {"collected": [state["item"]]}

def review(state):
    # This node may execute again on resume. No database writes before interrupt.
    approved = interrupt({"items": sorted(state["collected"])})
    return {"approved": bool(approved)}

def finish(state):
    return {"output": sorted(state["collected"]) if state["approved"] else []}

def build(saver):
    graph=StateGraph(State)
    graph.add_node("collect",collect);graph.add_node("review",review);graph.add_node("finish",finish)
    graph.add_conditional_edges(START,lambda s:[Send("collect",{"item":item}) for item in s["items"]])
    graph.add_edge("collect","review");graph.add_edge("review","finish");graph.add_edge("finish",END)
    return graph.compile(checkpointer=saver)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,required=True);parser.add_argument("--thread-id",default="lab-review-1")
    parser.add_argument("--phase",choices=("start","resume"),required=True)
    args=parser.parse_args();args.db.parent.mkdir(parents=True,exist_ok=True)
    cfg={"configurable":{"thread_id":args.thread_id}}
    with sqlite3.connect(args.db,check_same_thread=False) as connection:
        graph=build(SqliteSaver(connection))
        before=graph.get_state(cfg)
        if args.phase=="start":
            if before.values: raise ValueError("Use a fresh thread for a new lab")
            graph.invoke({"items":["A","B"],"collected":[],"output":[]},cfg)
        elif not before.values:
            raise ValueError("No checkpoint for this thread")
        elif before.next:
            graph.invoke(Command(resume=True),cfg)
        after=graph.get_state(cfg)
        print(json.dumps({"scope":"deterministic teaching lab, not model evidence","thread_id":args.thread_id,
            "next":list(after.next),"collected":sorted(after.values.get("collected",[])),
            "output":after.values.get("output",[]),"approved":after.values.get("approved")}))
    return 0

if __name__=="__main__":raise SystemExit(main())
