"""Model benchmark for CV/JD matching (experiments A and B).

  uv run python scripts/benchmark.py extract  --models qwen3.8-27b [--limit N]
  uv run python scripts/benchmark.py parse    --models qwen3.8-27b
  uv run python scripts/benchmark.py retrieve --models qwen3.8-27b [--embedding-model gemini-embedding-2]
  uv run python scripts/benchmark.py embed    --models text-embedding-3-small [--extractions-from M]
  uv run python scripts/benchmark.py report

Every model call goes to <out>/cache.jsonl keyed by (task, model, sha256(input));
re-runs skip cached items (errors too, unless --retry-errors). App config is read
at import, so each model runs in a fresh subprocess with CHAT_MODEL /
EMBEDDING_MODEL / CHROMA_COLLECTION set. The app's DB/data/Chroma paths are
pointed into <out>; Chroma is never used (retrieval is in-memory cosine).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_metrics as bm  # noqa: E402

CHAT_MODELS = ["gemini-3.5-flash-lite", "deepseek-flash", "gpt-6-luna", "qwen3.8-27b"]
EMBED_MODELS = ["gemini-embedding-2", "text-embedding-3-small", "text-embedding-3-large"]
DEFAULT_EMBEDDING = "gemini-embedding-2"
DEFAULT_SOURCE_CHAT = "gemini-3.5-flash-lite"
QUERY_MAX_CHARS = 3000  # schemas.SearchRequest.query max_length: the app rejects longer queries
RETRIEVE_K = 20         # config.RETRIEVE_K
STOP_AFTER_RATE_LIMITS = 3  # consecutive 429s: likely daily quota, stop instead of caching more errors
# Transient errors (Gemini 503 overload, 429, timeouts) are retried with backoff
# before being recorded; the number of retries is kept as a reliability metric.
TRANSIENT_ERRORS = ("rate_limit", "timeout")
RETRY_BACKOFF_S = (15, 30, 60)
EMBED_RETRY_BACKOFF_S = (60, 60, 90)
NOTES = {"qwen3.8-27b": "self-hosted vLLM on one RTX 5090; no API cost"}


# ---- Environment isolation (before any src.backend import) ----

def isolation_env(out: Path) -> dict:
    sandbox = out / "app_sandbox"
    return {"DATA_DIR": str(sandbox), "APP_DB_PATH": str(sandbox / "app.db"),
            "CHECKPOINT_DB_PATH": str(sandbox / "checkpoints.db"), "CHROMA_DIR": str(sandbox / "chroma")}


def model_env(out: Path, chat_model: str, embedding_model: str, dims: int) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("GEMINI_CHAT_MODEL", "GEMINI_EMBEDDING_MODEL", "GEMINI_EMBEDDING_DIMENSIONS")}
    env.update(isolation_env(out))
    env.update(CHAT_MODEL=chat_model, EMBEDDING_MODEL=embedding_model, EMBEDDING_DIMENSIONS=str(dims),
               CHROMA_COLLECTION=f"cv_{embedding_model.replace('-', '_')}_{dims}_v1",
               PYTHONPATH=str(ROOT))
    return env


def app_modules():
    from src.backend import config, services
    from src.backend.graphs.search import retrieval_text
    from src.backend.text_cleaning import prepare_llm_input
    return config, services, retrieval_text, prepare_llm_input


# ---- Data ----

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_data(data: Path, limit: int | None) -> dict:
    cvs, jds = read_jsonl(data / "cvs.jsonl"), read_jsonl(data / "jds.jsonl")
    if limit:
        cvs, jds = cvs[:limit], jds[:limit]
    gold = {row["cv_id"]: row for row in read_jsonl(data / "gold_extract.jsonl")}
    return {"cvs": cvs, "jds": jds, "gold": gold}


# ---- Model inputs (exactly what the app would send) ----

def extract_payload(text: str) -> dict:
    """Same shape as services.extract_cv builds after graphs.ingest.preprocess (first attempt)."""
    config, _, _, prepare_llm_input = app_modules()
    return {"cv": prepare_llm_input(text, config.LLM_INPUT_MAX_CHARS), "fix": []}


def jd_query(text: str) -> str:
    return " ".join(text.split())[:QUERY_MAX_CHARS].strip()


def extract_items(data) -> list[dict]:
    return [{"id": cv["cv_id"], "input": extract_payload(cv["text"])} for cv in data["cvs"]]


def parse_items(data) -> list[dict]:
    return [{"id": jd["jd_id"], "input": jd_query(jd["text"])} for jd in data["jds"]]


def extractions(cache, model: str, data) -> dict:
    """cv_id -> cache record for that chat model's extraction (missing = not run yet)."""
    out = {}
    for item in extract_items(data):
        record = cache.get("extract", model, bm.input_sha(item["input"]))
        if record:
            out[item["id"]] = record
    return out


def doc_items(cache, chat_model: str, data) -> list[dict]:
    _, services, _, _ = app_modules()
    return [{"id": cv_id, "input": services.embedding_text(rec["output"])}
            for cv_id, rec in extractions(cache, chat_model, data).items() if rec.get("ok")]


def query_items(cache, data, chat_model: str | None, query_text: str) -> list[dict]:
    """JD text as the vector query (default), or the app's criteria-shaped retrieval text."""
    if query_text == "jd" or chat_model is None:
        return [{"id": item["id"], "input": item["input"]} for item in parse_items(data)]
    _, _, retrieval_text, _ = app_modules()
    out = []
    for item in parse_items(data):
        rec = cache.get("parse", chat_model, bm.input_sha(item["input"]))
        criteria = rec["output"] if rec and rec.get("ok") else {}
        out.append({"id": item["id"], "input": retrieval_text(criteria, item["input"])})
    return out


# ---- Worker (runs inside a subprocess with one model configured) ----

def run_worker(args) -> int:
    from src.backend import config, services
    from src.backend.schemas import CVExtract
    cache = bm.CallCache(args.cache)
    items = json.loads(Path(args.items).read_text(encoding="utf-8"))
    model = config.CHAT_MODEL if args.task in ("extract", "parse") else config.EMBEDDING_MODEL
    assert model == args.model, f"configured {model} != requested {args.model}"
    rate_limited = 0
    if args.task in ("extract", "parse"):
        for n, item in enumerate(items, 1):
            for retries in range(len(RETRY_BACKOFF_S) + 1):
                services.last_call_stats.set(None)
                start, output, error, detail = time.perf_counter(), None, None, None
                try:
                    if args.task == "extract":
                        output = services.cloud_json(CVExtract, services.EXTRACT_INSTRUCTION,
                                                     item["input"]).to_record()
                    else:
                        output = services.parse_query(item["input"]).model_dump()
                except Exception as exc:  # every failure is a data point
                    error, detail = bm.classify_error(exc), f"{type(exc).__name__}: {exc}"[:300]
                elapsed = time.perf_counter() - start
                if error not in TRANSIENT_ERRORS or retries == len(RETRY_BACKOFF_S):
                    break
                print(f"[{model}] {item['id']} {error} ({detail}); retry in {RETRY_BACKOFF_S[retries]}s",
                      file=sys.stderr, flush=True)
                time.sleep(RETRY_BACKOFF_S[retries])
            stats = services.last_call_stats.get() or {}
            wait = float(stats.get("wait_s") or 0.0)
            cache.append({"task": args.task, "model": model, "input_sha": bm.input_sha(item["input"]),
                          "item_id": item["id"], "ok": error is None, "error": error,
                          "error_detail": detail, "latency_s": round(elapsed - wait, 4),
                          "wait_s": round(wait, 4), "input_tokens": stats.get("input_tokens"),
                          "output_tokens": stats.get("output_tokens"), "tokens_estimated": False,
                          "retries": retries, "output": output, "ts": time.time()})
            print(f"[{model}] {args.task} {n}/{len(items)} {item['id']} "
                  f"{'ok' if error is None else error} {elapsed - wait:.2f}s (wait {wait:.1f}s)",
                  file=sys.stderr, flush=True)
            # A dropped tunnel/server gives connection errors on every item: stop early too.
            rate_limited = rate_limited + 1 if error in ("rate_limit", "connection") else 0
            if rate_limited >= STOP_AFTER_RATE_LIMITS:
                print(f"[{model}] {STOP_AFTER_RATE_LIMITS} rate-limit/connection errors in a row; stopping "
                      "(re-run later, cached items are skipped)", file=sys.stderr)
                return 3
        return 0
    emb = services.embeddings()
    batch_size = 1 if args.task == "embed_query" else config.EMBED_BATCH_SIZE
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [item["input"] for item in batch]
        # Gemini free tier counts every text in a batch toward its per-minute quota,
        # so a 429 here is retried after a full minute instead of being recorded.
        for retries in range(len(EMBED_RETRY_BACKOFF_S) + 1):
            start, vectors, error, detail = time.perf_counter(), None, None, None
            emb.last_usage = None
            try:
                vectors = ([emb.embed_query(texts[0])] if args.task == "embed_query"
                           else emb.embed_documents(texts))
            except Exception as exc:
                error, detail = bm.classify_error(exc), f"{type(exc).__name__}: {exc}"[:300]
            elapsed, batch_ts = time.perf_counter() - start, time.time()
            if error not in TRANSIENT_ERRORS or retries == len(EMBED_RETRY_BACKOFF_S):
                break
            print(f"[{model}] {args.task} batch {i // batch_size + 1} {error}; retry in "
                  f"{EMBED_RETRY_BACKOFF_S[retries]}s", file=sys.stderr, flush=True)
            time.sleep(EMBED_RETRY_BACKOFF_S[retries])
        usage = getattr(emb, "last_usage", None) or {}
        if usage.get("prompt_tokens") is not None:
            tokens, estimated = bm.split_tokens(int(usage["prompt_tokens"]), texts), False
        else:
            tokens, estimated = [bm.estimate_tokens(t) for t in texts], True
        for j, item in enumerate(batch):
            cache.append({"task": args.task, "model": model, "input_sha": bm.input_sha(item["input"]),
                          "item_id": item["id"], "ok": error is None, "error": error,
                          "error_detail": detail, "latency_s": round(elapsed / len(batch), 4),
                          "batch_latency_s": round(elapsed, 4), "batch_size": len(batch), "wait_s": 0.0,
                          "input_tokens": tokens[j], "output_tokens": 0, "tokens_estimated": estimated,
                          "output": [round(v, 6) for v in vectors[j]] if vectors else None,
                          "batch_ts": batch_ts, "retries": retries, "ts": time.time()})
        print(f"[{model}] {args.task} {i + len(batch)}/{len(items)} "
              f"{'ok' if error is None else error} {elapsed:.2f}s", file=sys.stderr, flush=True)
        rate_limited = rate_limited + 1 if error == "rate_limit" else 0
        if rate_limited >= STOP_AFTER_RATE_LIMITS:
            print(f"[{model}] rate limited; stopping", file=sys.stderr)
            return 3
    return 0


def run_pending(args, task: str, model: str, items: list[dict]) -> None:
    """Run uncached items for one model in a fresh interpreter with that model configured."""
    cache = bm.CallCache(args.out / "cache.jsonl")
    pending = cache.pending(task, model, items, args.retry_errors)
    print(f"{task} {model}: {len(items) - len(pending)} cached, {len(pending)} to run", file=sys.stderr)
    if not pending:
        return
    config, *_ = app_modules()
    if task in ("extract", "parse"):
        chat, emb = model, DEFAULT_EMBEDDING
    else:
        chat, emb = DEFAULT_SOURCE_CHAT, model
    dims = config.EMBEDDING_MODELS[emb]["dimensions"]
    items_file = args.out / "tmp" / f"{task}_{model}.json"
    items_file.parent.mkdir(parents=True, exist_ok=True)
    items_file.write_text(json.dumps(pending, ensure_ascii=False), encoding="utf-8")
    code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "_worker", "--task", task,
                            "--model", model, "--items", str(items_file),
                            "--cache", str(args.out / "cache.jsonl")],
                           env=model_env(args.out, chat, emb, dims), cwd=ROOT)
    if code:
        print(f"worker {task} {model} exited with {code}", file=sys.stderr)


# ---- Metrics per experiment ----

def call_stats(records: list[dict], price=None, embed: bool = False) -> dict:
    lat = [r.get("latency_s") for r in records]
    costs = [bm.embed_cost(r.get("input_tokens"), price) if embed else
             bm.chat_cost(r.get("input_tokens"), r.get("output_tokens"), price) for r in records]
    total = bm.sum_known(costs)
    return {"calls": len(records), "latency_p50_s": bm.percentile(lat, 50),
            "latency_p95_s": bm.percentile(lat, 95),
            "wait_total_s": sum(r.get("wait_s") or 0 for r in records),
            "input_tokens_mean": bm.mean([r.get("input_tokens") for r in records]),
            "output_tokens_mean": bm.mean([r.get("output_tokens") for r in records]),
            "cost_usd_total": total,
            "cost_usd_per_1k_calls": total / len(records) * 1000 if total is not None and records else None,
            "tokens_estimated": any(r.get("tokens_estimated") for r in records)}


def error_counts(records: list[dict]) -> dict:
    ok = sum(bool(r.get("ok")) for r in records)
    by_type = {}
    for r in records:
        if not r.get("ok"):
            by_type[r["error"]] = by_type.get(r["error"], 0) + 1
    return {"valid_rate": ok / len(records) if records else None,
            "json_invalid": by_type.pop("json_invalid", 0),
            "schema_invalid": by_type.pop("schema_invalid", 0),
            "rate_limit": by_type.pop("rate_limit", 0), "timeout": by_type.pop("timeout", 0),
            "other_errors": json.dumps(by_type) if by_type else ""}


def extract_metrics(cache, model: str, data) -> dict | None:
    config, *_ = app_modules()
    recs = extractions(cache, model, data)
    if not recs:
        return None
    strict, lenient, years = [], [], []
    for cv_id, gold in data["gold"].items():
        if cv_id not in recs:
            continue
        out = recs[cv_id]["output"] if recs[cv_id].get("ok") else {}
        strict.append(bm.skill_prf(out.get("skills"), gold.get("skills")))
        lenient.append(bm.skill_prf(out.get("skills"), gold.get("skills"), lenient=True))
        years.append((out.get("years_experience"), gold.get("years_experience")))
    sp, sr, sf = bm.mean_prf(strict)
    lp, lr, lf = bm.mean_prf(lenient)
    ym = bm.years_metrics(years)
    records = list(recs.values())
    return {"experiment": "A_extract", "model": model, "items": len(data["cvs"]), "n_run": len(records),
            **error_counts(records), "n_gold": len(strict),
            "skill_p": sp, "skill_r": sr, "skill_f1": sf,
            "skill_p_lenient": lp, "skill_r_lenient": lr, "skill_f1_lenient": lf,
            "years_mae": ym["mae"], "years_within_1": ym["within_1"], "years_n": ym["n"],
            "years_missing_rate": ym["missing_rate"],
            **call_stats(records, config.CLOUD_CHAT_MODELS[model]["price"]), "notes": NOTES.get(model, "")}


def parsed(cache, model: str, data) -> dict:
    out = {}
    for item in parse_items(data):
        rec = cache.get("parse", model, bm.input_sha(item["input"]))
        if rec:
            out[item["id"]] = rec
    return out


def parse_metrics(cache, model: str, data) -> dict | None:
    config, *_ = app_modules()
    recs = parsed(cache, model, data)
    if not recs:
        return None
    strict, lenient, years, level = [], [], [], []
    for jd in data["jds"]:
        if jd["jd_id"] not in recs:
            continue
        rec, gold = recs[jd["jd_id"]], jd.get("gold") or {}
        out = rec["output"] if rec.get("ok") else {}
        strict.append(bm.skill_prf(out.get("skills"), gold.get("skills")))
        lenient.append(bm.skill_prf(out.get("skills"), gold.get("skills"), lenient=True))
        years.append(bm.same_number(out.get("min_years"), gold.get("min_years")))
        level.append(out.get("level") == gold.get("level"))
    records = list(recs.values())
    return {"experiment": "A_parse", "model": model, "items": len(data["jds"]), "n_run": len(records),
            **error_counts(records),
            "skill_p": bm.mean_prf(strict)[0], "skill_r": bm.mean_prf(strict)[1],
            "skill_f1": bm.mean_prf(strict)[2], "skill_f1_lenient": bm.mean_prf(lenient)[2],
            "min_years_exact": bm.mean([float(x) for x in years]),
            "level_exact": bm.mean([float(x) for x in level]),
            **call_stats(records, config.CLOUD_CHAT_MODELS[model]["price"]), "notes": NOTES.get(model, "")}


def vectors_for(cache, task: str, model: str, items: list[dict]) -> dict:
    out = {}
    for item in items:
        rec = cache.get(task, model, bm.input_sha(item["input"]))
        if rec and rec.get("ok"):
            out[item["id"]] = rec["output"]
    return out


def ranking_metrics(rankings: dict, category: dict, jd_category: dict, prefix: str) -> dict:
    p5, p10, rr = [], [], []
    for jd_id, ranked in rankings.items():
        relevant = [category.get(cv_id) == jd_category[jd_id] for cv_id in ranked]
        p5.append(bm.precision_at_k(relevant, 5))
        p10.append(bm.precision_at_k(relevant, 10))
        rr.append(bm.reciprocal_rank(relevant))
    return {f"{prefix}p_at_5": bm.mean(p5), f"{prefix}p_at_10": bm.mean(p10), f"{prefix}mrr": bm.mean(rr)}


def retrieve_metrics(cache, model: str, data, embedding_model: str, query_text: str) -> dict | None:
    docs = vectors_for(cache, "embed_doc", embedding_model, doc_items(cache, model, data))
    queries = vectors_for(cache, "embed_query", embedding_model,
                          query_items(cache, data, model, query_text))
    if not docs or not queries:
        return None
    recs = extractions(cache, model, data)
    crit = parsed(cache, model, data)
    category = {cv["cv_id"]: cv["category"] for cv in data["cvs"]}
    jd_category = {jd["jd_id"]: jd["category"] for jd in data["jds"]}
    jd_text = {item["id"]: item["input"] for item in parse_items(data)}
    from src.backend.scoring import score_cv
    vector_rank, rerank = {}, {}
    for jd_id, qvec in queries.items():
        ranked = bm.rank_by_cosine(qvec, docs)
        vector_rank[jd_id] = [cv_id for cv_id, _ in ranked]
        rec = crit.get(jd_id)
        # Same fallback as graphs.search.parse_query when parsing fails.
        criteria = (rec["output"] if rec and rec.get("ok") else
                    {"skills": [], "min_years": None, "level": None, "keywords": [jd_text[jd_id]]})
        # Chroma cosine distance d = 1 - cos; the app's similarity is 1 - d/2.
        scored = [score_cv(cv_id, recs[cv_id]["output"], criteria, 1 - (1 - cos) / 2)
                  for cv_id, cos in ranked[:RETRIEVE_K]]
        rerank[jd_id] = [r["cv_id"] for r in sorted(scored, key=lambda r: (-r["score"], r["cv_id"]))]
    return {"experiment": "A_retrieve", "model": model, "embedding_model": embedding_model,
            "query_text": query_text, "items": len(data["jds"]), "n_run": len(queries),
            "indexed_cvs": len(docs), "total_cvs": len(data["cvs"]),
            **ranking_metrics(vector_rank, category, jd_category, "vector_"),
            **ranking_metrics(rerank, category, jd_category, "rerank_"), "notes": NOTES.get(model, "")}


def embed_metrics(cache, model: str, data, source: str) -> dict | None:
    config, *_ = app_modules()
    ditems, qitems = doc_items(cache, source, data), query_items(cache, data, None, "jd")
    docs, queries = vectors_for(cache, "embed_doc", model, ditems), vectors_for(cache, "embed_query", model, qitems)
    if not docs or not queries:
        return None
    category = {cv["cv_id"]: cv["category"] for cv in data["cvs"]}
    jd_category = {jd["jd_id"]: jd["category"] for jd in data["jds"]}
    rankings = {jd_id: [cv_id for cv_id, _ in bm.rank_by_cosine(q, docs)] for jd_id, q in queries.items()}
    price = config.EMBEDDING_MODELS[model]["price"]
    drec = [r for it in ditems if (r := cache.get("embed_doc", model, bm.input_sha(it["input"])))]
    qrec = [r for it in qitems if (r := cache.get("embed_query", model, bm.input_sha(it["input"])))]
    allrec = drec + qrec
    total = bm.sum_known(bm.embed_cost(r.get("input_tokens"), price) for r in allrec)
    return {"experiment": "B_embed", "model": model, "extractions_from": source,
            "items": len(data["jds"]), "n_run": len(queries), "indexed_cvs": len(docs),
            "total_cvs": len(data["cvs"]), **error_counts(allrec),
            **ranking_metrics(rankings, category, jd_category, "vector_"),
            "query_latency_p50_s": bm.percentile([r.get("latency_s") for r in qrec], 50),
            "query_latency_p95_s": bm.percentile([r.get("latency_s") for r in qrec], 95),
            "doc_batch_latency_p50_s": bm.percentile(
                list({r.get("batch_ts"): r.get("batch_latency_s") for r in drec}.values()), 50),
            "input_tokens_total": sum(r.get("input_tokens") or 0 for r in allrec),
            "tokens_estimated": any(r.get("tokens_estimated") for r in allrec),
            "cost_usd_total": total,
            "cost_usd_per_1k_docs": (bm.sum_known(bm.embed_cost(r.get("input_tokens"), price) for r in drec)
                                     / len(drec) * 1000) if price is not None and drec else None,
            "notes": "" if price is not None else "price not published in config"}


# ---- Output ----

def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


TABLE_COLUMNS = {
    "A_extract": ["model", "n_run", "valid_rate", "json_invalid", "schema_invalid", "n_gold", "skill_f1",
                  "skill_f1_lenient", "years_mae", "years_within_1", "latency_p50_s", "latency_p95_s",
                  "input_tokens_mean", "output_tokens_mean", "cost_usd_total", "cost_usd_per_1k_calls"],
    "A_parse": ["model", "n_run", "valid_rate", "skill_f1", "skill_f1_lenient", "min_years_exact",
                "level_exact", "latency_p50_s", "latency_p95_s", "cost_usd_total"],
    "A_retrieve": ["model", "embedding_model", "indexed_cvs", "vector_p_at_5", "vector_p_at_10",
                   "rerank_p_at_5", "rerank_p_at_10", "rerank_mrr"],
    "B_embed": ["model", "extractions_from", "indexed_cvs", "vector_p_at_5", "vector_p_at_10", "vector_mrr",
                "query_latency_p50_s", "query_latency_p95_s", "input_tokens_total", "tokens_estimated",
                "cost_usd_total", "cost_usd_per_1k_docs"],
}


def markdown(rows: list[dict]) -> str:
    parts = []
    for experiment, cols in TABLE_COLUMNS.items():
        group = [r for r in rows if r["experiment"] == experiment]
        if not group:
            continue
        parts.append(f"### {experiment}\n")
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("|" + "---|" * len(cols))
        parts += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in group]
        notes = {r["model"]: r["notes"] for r in group if r.get("notes")}
        parts += [f"\n- {m}: {n}" for m, n in notes.items()]
        parts.append("")
    return "\n".join(parts)


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def calls_rows(cache) -> list[dict]:
    config, *_ = app_modules()
    rows = []
    for r in cache.records.values():
        embed = r["task"].startswith("embed")
        price = (config.EMBEDDING_MODELS if embed else config.CLOUD_CHAT_MODELS).get(r["model"], {}).get("price")
        cost = (bm.embed_cost(r.get("input_tokens"), price) if embed else
                bm.chat_cost(r.get("input_tokens"), r.get("output_tokens"), price))
        rows.append({k: r.get(k) for k in ("task", "model", "item_id", "ok", "error", "latency_s", "wait_s",
                                           "batch_size", "input_tokens", "output_tokens", "tokens_estimated",
                                           "input_sha", "error_detail")} | {"cost_usd": cost})
    return rows


# ---- Commands ----

def chat_models(args) -> list[str]:
    return args.models or CHAT_MODELS


def command(args) -> None:
    config, *_ = app_modules()
    data = load_data(args.data, args.limit)
    if not data["cvs"] and args.cmd in ("extract", "retrieve", "embed"):
        sys.exit(f"no CVs in {args.data / 'cvs.jsonl'}")
    allowed = EMBED_MODELS if args.cmd == "embed" else list(config.CLOUD_CHAT_MODELS)
    for m in args.models or []:
        if args.cmd != "report" and m not in allowed:
            sys.exit(f"unknown model {m}; choose from {allowed}")
    rows = []
    if args.cmd == "extract":
        for m in chat_models(args):
            run_pending(args, "extract", m, extract_items(data))
        cache = bm.CallCache(args.out / "cache.jsonl")
        rows = [extract_metrics(cache, m, data) for m in chat_models(args)]
    elif args.cmd == "parse":
        for m in chat_models(args):
            run_pending(args, "parse", m, parse_items(data))
        cache = bm.CallCache(args.out / "cache.jsonl")
        rows = [parse_metrics(cache, m, data) for m in chat_models(args)]
    elif args.cmd == "retrieve":
        for m in chat_models(args):
            run_pending(args, "extract", m, extract_items(data))
            run_pending(args, "parse", m, parse_items(data))
            cache = bm.CallCache(args.out / "cache.jsonl")
            run_pending(args, "embed_doc", args.embedding_model, doc_items(cache, m, data))
            run_pending(args, "embed_query", args.embedding_model,
                        query_items(cache, data, m, args.query_text))
        cache = bm.CallCache(args.out / "cache.jsonl")
        rows = [retrieve_metrics(cache, m, data, args.embedding_model, args.query_text)
                for m in chat_models(args)]
    elif args.cmd == "embed":
        cache = bm.CallCache(args.out / "cache.jsonl")
        ditems = doc_items(cache, args.extractions_from, data)
        if not ditems:
            sys.exit(f"no extractions cached for {args.extractions_from}; run "
                     f"`extract --models {args.extractions_from}` first")
        if len(ditems) < len(data["cvs"]):
            print(f"warning: only {len(ditems)}/{len(data['cvs'])} CVs have a valid "
                  f"{args.extractions_from} extraction", file=sys.stderr)
        for m in args.models or EMBED_MODELS:
            run_pending(args, "embed_doc", m, ditems)
            run_pending(args, "embed_query", m, query_items(cache, data, None, "jd"))
        cache = bm.CallCache(args.out / "cache.jsonl")
        rows = [embed_metrics(cache, m, data, args.extractions_from) for m in args.models or EMBED_MODELS]
    elif args.cmd == "report":
        cache = bm.CallCache(args.out / "cache.jsonl")
        seen = {(r["task"], r["model"]) for r in cache.records.values()}
        chats = [m for m in config.CLOUD_CHAT_MODELS if ("extract", m) in seen or ("parse", m) in seen]
        embeds = [m for m in config.EMBEDDING_MODELS if ("embed_doc", m) in seen]
        rows = ([extract_metrics(cache, m, data) for m in chats]
                + [parse_metrics(cache, m, data) for m in chats]
                + [retrieve_metrics(cache, m, data, args.embedding_model, args.query_text) for m in chats]
                + [embed_metrics(cache, m, data, args.extractions_from) for m in embeds])
        rows = [r for r in rows if r]
        if embeds and not any(r["experiment"] == "B_embed" for r in rows):
            print(f"note: no B_embed rows for extractions from {args.extractions_from}; "
                  "pass --extractions-from <chat model>", file=sys.stderr)
        write_csv(args.out / "summary.csv", rows)
        write_csv(args.out / "calls.csv", calls_rows(cache))
        print(f"wrote {args.out / 'summary.csv'} and {args.out / 'calls.csv'}", file=sys.stderr)
    print(markdown([r for r in rows if r]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["extract", "parse", "embed", "retrieve", "report", "_worker"])
    parser.add_argument("--models", nargs="+", help="chat models (extract/parse/retrieve) or "
                        "embedding models (embed); default: all benchmarked models")
    parser.add_argument("--limit", type=int, help="use only the first N CVs and first N JDs")
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmark")
    parser.add_argument("--out", type=Path, default=ROOT / "results/benchmark")
    parser.add_argument("--retry-errors", action="store_true", help="re-run cached failed calls")
    parser.add_argument("--extractions-from", default=DEFAULT_SOURCE_CHAT,
                        help="embed: chat model whose extractions are embedded")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING, help="retrieve: embedding model")
    parser.add_argument("--query-text", choices=["jd", "criteria"], default="jd",
                        help="retrieve: embed the JD text (default) or the app's criteria-shaped "
                             "retrieval_text built from that model's parse_query output")
    # internal (_worker)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--items", help=argparse.SUPPRESS)
    parser.add_argument("--cache", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "_worker":
        return run_worker(args)
    args.out, args.data = args.out.resolve(), args.data.resolve()
    # Never touch the app's data: isolate before src.backend is imported.
    os.environ.update(isolation_env(args.out))
    command(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
