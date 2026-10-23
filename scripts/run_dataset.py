"""Resume real CV extraction/indexing in bounded batches, then exercise semantic search."""

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

from scripts.redact_contacts import REDACTION_VERSION, redact_contacts


def prepare_ingest_item(item):
    """Only the filtered CV text may cross the ingest/model boundary."""
    text, counts = redact_contacts(item["raw_text"])
    payload = {"cv_id": item["cv_id"], "file_name": item["file_name"], "file_path": "",
               "raw_text": text, "display_name": item["display_name"]}
    return payload, counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("data/datasets/djinni/sample-100000.jsonl"))
    parser.add_argument("--limit",type=int,default=100,help="Pilot size; use 100000 for the full corpus")
    parser.add_argument("--batch-size",type=int,default=20)
    parser.add_argument("--query",action="append",default=[])
    parser.add_argument("--report",type=Path,default=Path("data/benchmarks/gemini-dataset-run.json"))
    args=parser.parse_args()
    if args.limit < 1 or not 1 <= args.batch_size <= 20:
        parser.error("limit must be positive and batch-size must be 1..20")
    manifest_path=args.input.with_suffix(".manifest.json")
    if not args.input.is_file() or not manifest_path.is_file():
        print(json.dumps({"status":"blocked","error":"Prepared JSONL and manifest are required."}))
        return 2
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    digest=hashlib.sha256()
    with args.input.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(chunk)
    if digest.hexdigest()!=manifest.get("output_sha256"):
        print(json.dumps({"status":"blocked","error":"Prepared dataset checksum mismatch."}))
        return 2
    required=("dotenv","langgraph","langchain_google_genai","chromadb","fastapi")
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print(json.dumps({"status":"blocked","missing_dependencies":missing,
                          "instruction":"Run uv sync --extra research first."}))
        return 2
    try:
        socket.getaddrinfo("generativelanguage.googleapis.com",443)
    except OSError as exc:
        print(json.dumps({"status":"blocked","error":"Gemini DNS: "+str(exc),"processed":0}))
        return 2
    from src.backend import db,services
    from src.backend.graphs.ingest import run_ingest
    from src.backend.routers.searches import create_search
    from src.backend.schemas import SearchRequest
    services.require_gemini()
    db.init_db()
    report={"status":"running","input":str(args.input),"requested":args.limit,
            "processed":0,"ready":0,"failed":0,"already_ready":0,"searches":[],
            "models":services.model_settings(),
            "contact_redaction":{"version":REDACTION_VERSION,"profiles_with_redactions":0,
                                 "replacement_counts":{}}}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def save_report():
        partial=args.report.with_suffix(".json.part")
        partial.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        partial.replace(args.report)
    redaction_counts=Counter()
    batch=[]; seen=0; consecutive_failed_batches=0; started=perf_counter()
    def ingest_batch(items):
        nonlocal consecutive_failed_batches
        outcomes=run_ingest(items)
        good=sum(item["status"]=="ready" for item in outcomes)
        report["processed"]+=len(items)
        report["ready"]+=good
        report["failed"]+=len(items)-good
        consecutive_failed_batches=0 if good else consecutive_failed_batches+1
        report["elapsed_seconds"]=round(perf_counter()-started,3)
        save_report()
        print(json.dumps({key:value for key,value in report.items() if key!='searches'}),flush=True)
        if consecutive_failed_batches>=3:
            raise RuntimeError("Three batches failed entirely; inspect provider access/quota and CV errors before retrying.")
    try:
        with args.input.open(encoding="utf-8") as handle:
            for line in handle:
                if seen>=args.limit:
                    break
                item=json.loads(line); seen+=1
                payload, counts=prepare_ingest_item(item)
                if counts:
                    report["contact_redaction"]["profiles_with_redactions"]+=1
                    redaction_counts.update(counts)
                    report["contact_redaction"]["replacement_counts"]=dict(redaction_counts)
                existing=db.get_cv(item["cv_id"])
                # Re-index an older row if it still contains unfiltered text.
                if existing and existing["status"]=="ready" and existing.get("raw_text")==payload["raw_text"] and services.has_vector(item["cv_id"]):
                    report["already_ready"]+=1
                    continue
                batch.append(payload)
                if len(batch)==args.batch_size:
                    ingest_batch(batch); batch=[]
            if batch:
                ingest_batch(batch)
        for query in args.query:
            response=create_search(SearchRequest(query=query,top_k=10))
            report["searches"].append({"query":query,"search_id":response["id"],"status":response["status"],
                                        "shortlist_count":len(response["shortlist"])})
        report["status"]="completed"
    except Exception as exc:
        report["status"]="stopped"
        report["error_type"]=type(exc).__name__
        # Avoid persisting arbitrary provider exception text, headers or credentials.
        print(json.dumps({"status":"stopped","error_type":type(exc).__name__}),file=sys.stderr)
    report["elapsed_seconds"]=round(perf_counter()-started,3)
    save_report()
    return 0 if report["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
