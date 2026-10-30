# JD Matching

HR uploads CV PDFs, searches them with free text, reviews a shortlist, and gets interview questions for approved candidates. FastAPI serves the API and plain HTML/JS/CSS frontend. Workflows use LangGraph, Google Gemini and a persistent Chroma index.

The implementation follows sections 11–12 of the [project report](https://docs.google.com/document/d/1plqpN92O382PiMdJRfrdgRB-tPgl-N51V9so5R_y0xk/edit?tab=t.0).

## Run

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Keys stay on the server. No provider requests are made at startup. Health, stored CVs and search history remain available without a key; upload and new search return HTTP 503 until a key is configured.

### Docker

```bash
docker compose up -d --build
```

Open http://localhost:8000; API docs: http://localhost:8000/docs. Stop with `docker compose down`. Data is persisted in `./data`.

### Local

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run uvicorn src.backend.main:app --reload
```

`uv sync` resolves the new dependencies and refreshes `uv.lock`; commit the refreshed lock after resolution. The Docker build also allows lock refresh when project metadata changes.

## Implemented workflows

- **Ingest:** two phases per bounded input batch. Phase 1 uses `Send(process_cv)` for up to four PDF/text workers: regex contact redaction, whitespace cleanup, one paced Gemini extraction, validation, and a committed SQLite extraction cache. A 429/timeout switches immediately to the pinned Ollama model; invalid output fails the record without another cloud call. A LangGraph barrier waits for every phase-1 branch before phase 2 embeds groups of up to 20 compact profiles in one request and bulk upserts them into Chroma. Each ready checkpoint is committed individually. Failed embedding runs reuse the extraction cache on restart. Threads and processes sharing `app.db` serialize ingest phases with an OS file lock; all application chat calls share a SQLite 12-RPM pacer.
- **Search:** parse a free-text query into criteria, retrieve up to 20 vectors, score candidates in parallel, rank by descending score then ascending CV ID, retain `top_k`, and explain only scores at least 0.5. Invalid query extraction falls back to semantic search. Stale/failed CVs are omitted; an explanation error leaves `reasons=null` and preserves ranking. Retrieval failure marks the search failed.
- **Review:** the graph calls `interrupt()` and persists its state using `SqliteSaver`. `search_id` is the checkpoint `thread_id`. Resuming with `Command(resume=...)` generates questions only for approved shortlist IDs, across technical, experience and behavioral categories. Empty approval ends without questions. Each candidate's question-generation failure is isolated.
- **Recovery:** waiting searches survive process restarts when `app.db` and `checkpoints.db` are retained. Review claims are atomic; repeated or concurrent submissions return 409. If graph completion succeeds but writing business results fails, retrying the same decision reuses the checkpoint questions. Keep the same decision when retrying.
- **Interface:** upload multiple files, filter CV status, open CV details, inspect matched/missing skills and explanations, approve candidates, view questions and reopen saved searches through history or a `#search=<id>` URL.
- **Delete:** removes the vector, uploaded PDF and CV row. Historical search records remain available.

The score is `0.5 * skill_fit + 0.3 * similarity + 0.2 * experience_fit`. Skill names are trimmed and compared case-insensitively. No required skills gives `skill_fit=1`. Missing experience contributes 0 only when a positive minimum is requested; it remains unknown in the CV record. No minimum, or a minimum of 0, gives `experience_fit=1`.

Cosine distance is normalized as `similarity = clamp(1 - distance / 2, 0, 1)`. Candidates must exceed `MIN_SIMILARITY` (default 0.55). This threshold and the ranking weights are experimental settings from the report, not calibrated hiring probabilities. HR makes the review decision.

## Configuration

All paths are relative to the working directory unless absolute.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | unset | Required for new ingest/search operations |
| `GEMINI_CHAT_MODEL` | `gemini-3.5-flash-lite` | Structured extraction, criteria, explanations, questions |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Same model for index and queries |
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | Fixed vector size; other values rejected |
| `GEMINI_TIMEOUT_SECONDS` | `60` | Timeout per provider request |
| `GEMINI_MAX_RETRIES` | `1` | Embedding SDK retries only; chat SDK attempts fixed to 1; 429 handled separately |
| `GEMINI_MAX_RPM` | `12` | Shared chat pacer, at least 5 s between admissions; only 1..12 accepted |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Only permitted local model |
| `OLLAMA_KEEP_ALIVE` | `30m` | Only `30m` or `-1` accepted |
| `OLLAMA_NUM_CTX` | `4096` | Fixed context size |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local fallback endpoint |
| `EMBED_BATCH_SIZE` | `20` | Phase-2 group size, 1..20 |
| `DATA_DIR` | `data` | Application data directory |
| `APP_DB_PATH` | `$DATA_DIR/app.db` | Business records |
| `CHECKPOINT_DB_PATH` | `$DATA_DIR/checkpoints.db` | Durable LangGraph state |
| `CHROMA_DIR` | `$DATA_DIR/chroma` | Persistent vector index |
| `CHROMA_COLLECTION` | `cv_gemini_embedding_2_768_v1` | Collection with embedding-model metadata |
| `MIN_SIMILARITY` | `0.55` | Experimental normalized-similarity cutoff |

This baseline rejects other chat/embedding models, dimensions and collection names at startup. Records created by the old `change me` stubs do not have real text or vectors and must be reuploaded. Never commit `.env` or data files.

Limits: 20 files per upload, 10 MB per file, 100 PDF pages, 24,000 extracted text characters, 3,000 query characters, `top_k` 1–20, graph recursion limit 25 and maximum concurrency 4. PDF scans need OCR, which is outside scope. Accounts and automatic hiring decisions are also outside scope.

Node metadata logs record IDs, execution time and error classes. Model logs record token counts when the provider supplies usage; unknown counts remain absent. CV content and keys are not included in these application logs. Configure Python logging at INFO to see metadata. LangSmith is optional via its standard environment configuration.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/cvs` | Upload PDFs, multipart field `files` |
| GET | `/cvs` | List CVs, optional status; `limit` 1–200 (default 100), `offset` >= 0 |
| GET | `/cvs/{id}` | CV detail |
| DELETE | `/cvs/{id}` | Delete CV and vector |
| POST | `/searches` | Start search and return shortlist |
| POST | `/searches/{id}/review` | Approve IDs and return questions; 409 unless waiting |
| GET | `/searches` | Search history |
| GET | `/searches/{id}` | Saved criteria, shortlist, approval and questions |
| GET | `/health` | `{ "status": "ok" }` |

## Tests

```bash
uv sync
uv run pytest -q
node --test tests/frontend.test.cjs
```

The suite uses actual SQLite, LangGraph, Chroma and PDF parsing, with deterministic doubles at the model/embedding boundaries. It covers the report scores (1.00 and 0.52), cloud-to-local failover and two-phase extraction-cache recovery, empty/stale candidates, failures, review filtering, duplicate submissions, independent threads, a new-process API resume, upload cleanup and PDF-to-review integration. It makes no Google Gemini requests and does not measure model quality.

Core scoring and SQLite transaction tests can also run with Python's standard library, without installing application dependencies:

```bash
python -m unittest discover -s tests -p test_scoring.py -v
python -m unittest discover -s tests -p test_database.py -v
```

## Real 100,000-profile corpus

### Corpus GUI

Open the **Corpus** tab to perform corpus operations without typing script commands:

- **Create sample:** uses the local `data/datasets/djinni/train.parquet`, filters valid unique English profiles and creates a new JSONL/manifest pair. Existing samples are preserved. Local setup needs `uv sync --locked --extra research`; Docker includes this dependency.
- **Verify:** checks the selected JSONL against its manifest checksum, without model requests.
- **Extract + embed:** choose 1–50 profiles per run and 1–20 per batch. The default is 20 profiles in batches of 5. The next-run mode remembers the position after each saved batch, skips matching ready rows with existing vectors, and advances to new work. To retry failures or revisit earlier records, use **Choose range** with a 1-based starting position; the limit then bounds the source range, including ready rows skipped within it. This does not change the forward cursor.
- **Benchmark:** runs the same FTS5/BM25 baseline as `scripts.benchmark_corpus`, with contact filtering, five query groups, p50/p95 and separate output files per run. No LLM/embedding requests are made.
- **Track jobs:** progress, ready/failed/skipped counts, graceful stop after the current batch, saved history and downloadable JSON reports. Results and cursors persist in `data/corpus_jobs.db` alongside `app.db`; interrupted server jobs become resumable via a new run after their 60-second lease expires. Only one corpus job runs at a time across server workers.

Prepared samples are discovered in the corpus directory; preview pagination reads only a page into the browser and masks recognizable contacts. Ready/failed overview cards count all application CVs, not a claim that the entire selected corpus was processed. Opening the tab or starting the server does not invoke a model. Ingest invokes the existing Gemini/Ollama extraction and Gemini embedding pipeline only after clicking **Run extract + embed**. Search/review continue in their existing tabs.

Corpus endpoints: `GET /corpus`, `GET /corpus/{id}/preview`, `POST /corpus/jobs`, `GET /corpus/jobs/{id}`, `POST /corpus/jobs/{id}/stop`, and `GET /corpus/jobs/{id}/report`. Job requests accept allowlisted actions and bounded numeric settings, never shell commands or arbitrary paths. This remains a local demo app with the same authentication scope as the existing upload/search endpoints.

See [docs/DATASET.md](docs/DATASET.md) for the downloaded Djinni source, exact row counts,
checksums, reproducible sampling, measured local lexical baseline and the resumable Google Gemini run.
CV lists are paginated and exclude raw CV text from summary queries, so a 100,000-profile
corpus does not load all records into the browser. The corpus importer accepts actual CV text
and uses explicit anonymous display IDs when the source intentionally omits personal names.

The research importer masks recognizable emails, contact links, messenger handles and phone-like
sequences before extraction, embedding and storage in the application database. It records only
aggregate replacement counts. Original dataset files remain unchanged locally. This contact
filter does not guarantee full anonymization; see the limitations in `docs/DATASET.md`.

## Check real Google Gemini inference before importing CVs

From a normal terminal with working HTTPS access, run the following from the repository root:

```powershell
python -m scripts.check_gemini_model
```

This uses only Python's standard library, loads the existing `.env`, and attempts one short Gemini generateContent API request with the configured chat model. It sends no CVs. API quota and billing are managed through Google AI Studio / its Google Cloud project. The result is saved in `data/benchmarks/gemini-model-check.json`, including the returned model/version, response ID, provider usage, output and elapsed time when available. Only `status=succeeded` confirms a completed inference response; this check does not evaluate extraction or the CV workflow. HTTP error messages and keys are not logged.

The check honors `GEMINI_TIMEOUT_SECONDS` from `.env`. To try a slower connection once, use `python -m scripts.check_gemini_model --timeout 60`. A socket timeout does not identify whether the delay occurred during connection, TLS, or reading a response, and does not validate the API key. Check DNS, TCP port 443 and an HTTPS response before treating the failure as model latency. The checker makes one attempt and does not automatically retry a timed-out inference.

If curl HTTPS works but Python times out, run `python -m scripts.diagnose_gemini_connection`. It compares Python urllib and curl using the same HEAD and unauthenticated POST requests. It sends no API key or CVs; the POST is expected to return an authentication error, which only demonstrates HTTP reachability. The report in `data/benchmarks/gemini-connection-check.json` contains status codes, timings and proxy-presence flags, without proxy addresses, credentials or response bodies. Python urllib can discover Windows registry proxy settings as well as environment variables, so its route can differ from curl. The diagnostic does not change proxy settings, certificate verification or network configuration.

Once PyPI works, install the actual project environment using `data/tools/uv.exe --cache-dir data/uv-cache sync --upgrade --extra research`. Always run installed application commands through `data/tools/uv.exe run --no-sync` so they use the project virtual environment. To compare the official SDK with the stdlib check, run `data/tools/uv.exe run --no-sync python -m scripts.check_gemini_model --transport sdk --timeout 60`. The SDK check explicitly uses the Gemini Developer API, the same short prompt, a timeout in milliseconds and one total attempt without automatic retry. A metadata HTTP 200 is not proof that inference or the CV workflow succeeded. The dataset runner now uses the SDK check after installing dependencies and passing tests, before sending any CVs.

If the SDK reports HTTP 504 / DEADLINE_EXCEEDED, an API deadline was exceeded; it is not a completed model response. Google GenAI SDK 2.33 sets X-Server-Timeout from the configured timeout. The diagnostic also accepts only the pinned `gemini-3.5-flash-lite`; alternate model names are rejected before sending a request. Keep each attempt report separate with `--report`.

After that succeeds, run `.\scripts\run-real-data.ps1 -Limit 20` for the actual dataset pilot.

## Provider migration to Gemini

Create a key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY` in `.env`. `GOOGLE_API_KEY` is accepted as a fallback; an old OpenAI key is never used. The model check uses only Python's standard library and sends no CVs. Do not paste keys into reports or chat.

The application calls `google-genai` for native JSON chat and embeddings, and the Ollama HTTP API for local fallback. Both chat providers receive the same prompt and short-key JSON Schema and validate through the same Pydantic model. Gemini Embedding 2 uses different query/document prefixes. Explicit `Content` boundaries produce separate vectors in one `batchEmbedContents` request; no legacy `task_type` is sent. Embeddings are normalized and their dimensions checked. Other embedding models are rejected.

A new collection includes provider, model, dimension and format metadata. Old collections remain separate; incompatible metadata is rejected. The corpus runner skips ready records only when their text matches and their vector exists in the current collection. The threshold 0.55 remains experimental and must be recalibrated for Gemini embeddings.

`uv.lock` still requires regeneration with working PyPI access: run `uv sync --upgrade --extra research`. Until that succeeds, do not use `uv sync --locked` or claim dependency installation/integration is verified. Historical experiment logs keep the provider used at the time and are not relabelled as successful Gemini runs.


## Fixed dual-model ingest baseline

See [docs/INGEST_RULES.md](docs/INGEST_RULES.md) for the rule-by-rule audit, two-phase recovery contract and verification limits.

Install the only local fallback model once with `ollama pull qwen2.5:3b`, and keep the Ollama server running. The application never requests another local model. Each local generation sends `format="json"`, temperature 0, context 4096 and `keep_alive="30m"` (or `-1`). Local generations are serialized within each app process. Docker needs an Ollama URL reachable from inside the container, such as `http://host.docker.internal:11434` for an Ollama server on the Windows host.

Embedding rate limits use three waits of 10/20/40 seconds plus 1..3 seconds of random jitter. No local embedding fallback is used. Chat and embedding quotas remain provider-controlled and are not transferred between models. Keep `app.db`, `app.ingest.lock` (while running) and Chroma on the same host for coordination; separate databases do not share pacing or phase locks.

## Quyết định bàn giao

Điểm là xếp hạng ưu tiên mềm, không phải bộ lọc điều kiện bắt buộc. API/UI dùng outcome để phân biệt succeeded, completed_with_errors, failed và pending; status completed chỉ cho biết workflow đã kết thúc. Xem docs/DOI_CHIEU_BAN_GIAO.md và lab labs/state_checkpoint. Không mở ingest toàn corpus; mỗi pilot giữ 20 CV.
