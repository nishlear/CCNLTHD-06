"""Export LangGraph diagrams as Mermaid, and as PNG when mmdc is installed.

Rendering stays local: draw_mermaid_png() would send the graph to mermaid.ink.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from src.backend.graphs.ingest import build_ingest_graph, build_process_graph
from src.backend.graphs.search import build_search_graph

GRAPHS = {
    "ingest_phase1": build_process_graph,
    "ingest": build_ingest_graph,
    "search": lambda: build_search_graph(InMemorySaver()),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/report/graphs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mmdc = shutil.which("mmdc")
    for name, build in GRAPHS.items():
        mmd = out / f"{name}.mmd"
        mmd.write_text(build().get_graph().draw_mermaid(), encoding="utf-8")
        print(mmd)
        if mmdc:
            png = mmd.with_suffix(".png")
            subprocess.run([mmdc, "-q", "-i", str(mmd), "-o", str(png), "-s", "2", "-b", "white"], check=True)
            print(png)
    if not mmdc:
        print("mmdc not found; install @mermaid-js/mermaid-cli to render PNG")


if __name__ == "__main__":
    main()
