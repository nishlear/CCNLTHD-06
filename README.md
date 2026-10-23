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

- **Ingest:** `Send(process_cv)` processes up to four CVs concurrently. Each subgraph reads PDF text, extracts structured data, validates nonempty `full_name` and `skills`, and retries with missing-field feedback. There are at most **two extraction attempts total**, including the first. Valid CVs are embedded and indexed in Chroma; each file finishes `ready` or `failed` with a reason. An index failure preserves a failed SQLite row and the extracted information.
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
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | Vector size; changing it requires reindexing |
| `GEMINI_TIMEOUT_SECONDS` | `60` | Timeout per provider request |
| `GEMINI_MAX_RETRIES` | `1` | Provider retries, separate from graph attempts |
| `DATA_DIR` | `data` | Application data directory |
| `APP_DB_PATH` | `$DATA_DIR/app.db` | Business records |
| `CHECKPOINT_DB_PATH` | `$DATA_DIR/checkpoints.db` | Durable LangGraph state |
| `CHROMA_DIR` | `$DATA_DIR/chroma` | Persistent vector index |
| `CHROMA_COLLECTION` | `cv_gemini_embedding_2_768_v1` | Collection with embedding-model metadata |
| `MIN_SIMILARITY` | `0.55` | Experimental normalized-similarity cutoff |

Changing embedding model requires a new collection and reupload/reindex of CVs. Records created by the old `change me` stubs do not have real text or vectors and must be reuploaded. Never commit `.env` or data files.

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

The suite uses actual SQLite, LangGraph, Chroma and PDF parsing, with deterministic doubles at the model/embedding boundaries. It covers the report scores (1.00 and 0.52), bounded extraction retries, empty/stale candidates, failures, review filtering, duplicate submissions, independent threads, a new-process API resume, upload cleanup and PDF-to-review integration. It makes no Google Gemini requests and does not measure model quality.

Core scoring and SQLite transaction tests can also run with Python's standard library, without installing application dependencies:

```bash
python -m unittest discover -s tests -p test_scoring.py -v
python -m unittest discover -s tests -p test_database.py -v
```

## Real 100,000-profile corpus

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

If the SDK reports HTTP 504 / DEADLINE_EXCEEDED, an API deadline was exceeded; it is not a completed model response. Google GenAI SDK 2.33 sets X-Server-Timeout from the configured timeout. To compare a lower-latency supported model once, add `--model gemini-3.5-flash-lite --transport sdk --timeout 60` to the model check. This CLI override does not change the application model; only update `GEMINI_CHAT_MODEL` in `.env` after the alternative model succeeds. Keep each attempt report separate with `--report`.

After that succeeds, run `.\scripts\run-real-data.ps1 -Limit 100` for the actual dataset pilot.

## Provider migration to Gemini

Create a key at https://aistudio.google.com/apikey and set `GEMINI_API_KEY` in `.env`. `GOOGLE_API_KEY` is accepted as a fallback; an old OpenAI key is never used. The model check uses only Python's standard library and sends no CVs. Do not paste keys into reports or chat.

The application uses `langchain-google-genai>=4.4.2,<5` for native structured chat output and `google-genai>=2.20,<3` for embeddings. Gemini Embedding 2 uses different query/document prefixes and one request per text; no legacy `task_type` is sent to this model. Gemini Embedding 001 is also supported with the corresponding retrieval task types. Embeddings are normalized and their dimensions checked.

A new collection includes provider, model, dimension and format metadata. Old collections remain separate; incompatible metadata is rejected. The corpus runner skips ready records only when their text matches and their vector exists in the current collection. The threshold 0.55 remains experimental and must be recalibrated for Gemini embeddings.

`uv.lock` still requires regeneration with working PyPI access: run `uv sync --upgrade --extra research`. Until that succeeds, do not use `uv sync --locked` or claim dependency installation/integration is verified. Historical experiment logs keep the provider used at the time and are not relabelled as successful Gemini runs.
