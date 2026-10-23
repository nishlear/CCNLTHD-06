"""Measure an explicit SQLite FTS5 lexical baseline on real CVs, without model calls."""

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from pathlib import Path
from time import perf_counter
from collections import Counter

from scripts.redact_contacts import REDACTION_VERSION, redact_contacts

QUERIES = {
    "python_fastapi": '"python" AND "fastapi"',
    "java_spring": '"java" AND "spring"',
    "react_typescript": '"react" AND "typescript"',
    "docker_kubernetes": '"docker" AND "kubernetes"',
    "sql_data": '"sql" AND "data"',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("data/datasets/djinni/sample-100000.jsonl"))
    parser.add_argument("--database",type=Path,default=Path("data/benchmarks/djinni_100k_fts.sqlite"))
    parser.add_argument("--report",type=Path,default=Path("data/benchmarks/djinni_100k_fts.json"))
    parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--redact-contacts",action="store_true",help="Mask recognizable contacts before indexing; includes filter time in build time")
    args=parser.parse_args()
    if args.database.exists():
        parser.error("benchmark database already exists; choose another path")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.database.parent.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    conn=sqlite3.connect(args.database)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE VIRTUAL TABLE profiles USING fts5(cv_id UNINDEXED, position, raw_text, tokenize='unicode61')")
        batch=[]; count=0; digest=hashlib.sha256()
        redacted_profiles=0; redaction_counts=Counter(); redaction_seconds=0.0
        with args.input.open("rb") as handle:
            for line in handle:
                digest.update(line)
                row=json.loads(line)
                if args.redact_contacts:
                    before=perf_counter()
                    row["raw_text"],replacements=redact_contacts(row["raw_text"])
                    redaction_seconds+=perf_counter()-before
                    if replacements:
                        redacted_profiles+=1
                        redaction_counts.update(replacements)
                batch.append((row["cv_id"],row["position"],row["raw_text"]))
                if len(batch)==1000:
                    conn.executemany("INSERT INTO profiles VALUES (?,?,?)",batch)
                    count+=len(batch); batch=[]
                    if count % 20000 == 0:
                        print(json.dumps({"indexed":count}),flush=True)
        if batch:
            conn.executemany("INSERT INTO profiles VALUES (?,?,?)",batch)
            count+=len(batch)
        conn.commit()
        build_seconds=perf_counter()-started
        results=[]
        for name,query in QUERIES.items():
            total=conn.execute("SELECT count(*) FROM profiles WHERE profiles MATCH ?",(query,)).fetchone()[0]
            elapsed=[]; rows=[]
            for _ in range(args.repeats):
                before=perf_counter()
                rows=conn.execute("SELECT cv_id,position,bm25(profiles) FROM profiles "
                                  "WHERE profiles MATCH ? ORDER BY bm25(profiles) LIMIT 10",(query,)).fetchall()
                elapsed.append((perf_counter()-before)*1000)
            results.append({"query":query,"name":name,"matches":total,
                            "p50_ms":round(statistics.median(elapsed),3),
                            "p95_ms":round(sorted(elapsed)[math.ceil(.95*len(elapsed))-1],3),
                            "top10":[{"cv_id":row[0],"position":row[1],"bm25":row[2]} for row in rows]})
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    report={"method":"SQLite FTS5 lexical baseline", "records":count,
            "source_jsonl_sha256":digest.hexdigest(), "sqlite_version":sqlite3.sqlite_version,
            "build_seconds":round(build_seconds,3),"database_bytes":args.database.stat().st_size,
            "query_repeats":args.repeats,"warm_cache":True,"queries":results,
            "openai_requests":0,"embedding_requests":0,
            "indexed_text":"contact-filtered" if args.redact_contacts else "original",
            "contact_redaction": {"version":REDACTION_VERSION,"profiles_with_redactions":redacted_profiles,
                                  "replacement_counts":dict(redaction_counts),
                                  "seconds":round(redaction_seconds,3)} if args.redact_contacts else None,
            "build_includes_contact_filter":args.redact_contacts,
            "limitations":"This is a local lexical baseline, not Chroma ANN or an OpenAI/LangGraph "
                          "end-to-end run. No relevance labels: precision, recall and nDCG are not measured."}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({**report,"queries":[{k:v for k,v in result.items() if k!='top10'} for result in results]},ensure_ascii=False))


if __name__ == "__main__":
    main()
