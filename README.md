# JD Matching

A CV screening app built on LangGraph. HR uploads CVs, searches them with a free-text job description, reviews a ranked shortlist, and gets interview questions for the candidates they approve.

- **Backend:** FastAPI, LangGraph, SQLite (business data and checkpoints), Chroma (vectors).
- **Frontend:** plain HTML/JS/CSS, served by the backend.
- **Models:** one cloud chat model and one embedding model, each picked from an allowlist, plus a local Qwen model in Ollama as fallback.

## Quick start

1. Copy `.env.example` to `.env` and set the API key for the models you choose. The default models need only `GEMINI_API_KEY`.
2. Start the app with Docker or locally.

**Docker**

```bash
docker compose up -d --build
```

**Local** (Python 3.11+ and [uv](https://docs.astral.sh/uv/))

```bash
uv sync
uv run uvicorn src.backend.main:app --reload
```

Open http://localhost:8000. API docs are at http://localhost:8000/docs. Data is stored in `./data`.

The app makes no model calls at startup. Without a key, health, stored CVs and search history still work; upload and new searches return HTTP 503.

For the local fallback, install Ollama and run `ollama pull qwen2.5:3b`. With Docker, point `OLLAMA_BASE_URL` at a host the container can reach, e.g. `http://host.docker.internal:11434`.

## How it works

### Ingest graph

Ingest runs in two phases for each batch of CVs.

1. **Extract.** `Send` starts one `process_cv` branch per CV (at most 4 at a time). Each branch reads the PDF or text, removes contact details, calls the chat model for structured JSON, validates it, and commits the result to an extraction cache in SQLite.
2. **Embed.** A barrier waits for every branch. Then compact profiles are embedded in groups of up to 20 and bulk upserted into Chroma. Each CV is marked `ready` as soon as it is saved.

If the cloud model returns 429, 503 or times out, extraction switches to the local Ollama model. If the JSON is valid but required fields are missing, extraction is retried (`MAX_EXTRACT_ATTEMPTS`, default 2). If embedding fails, the next run reuses the extraction cache instead of calling the chat model again.

### Search graph

`parse_query → retrieve → score_cv → rank → explain → human_review → generate_questions`

- `parse_query` turns the query into criteria: skills, minimum years, level and keywords. If parsing fails, the raw query is used as a keyword.
- `retrieve` builds a vector query from the criteria and fetches up to 20 candidates above `MIN_SIMILARITY`.
- `score_cv` runs once per candidate, in parallel:

  ```
  score = 0.5 * skill_fit + 0.3 * similarity + 0.2 * experience_fit
  similarity = clamp(1 - cosine_distance / 2, 0, 1)
  ```

  - **skill_fit:** the share of required skills found in the CV. Matching is case-insensitive.
  - **experience_fit:** `min(cv_years / min_years, 1)`. Unknown years score 0. If the query sets no minimum, it is 1.
  - The score ranks candidates. It does not filter them. A high score does not mean the candidate meets every requirement.
- `rank` sorts by score, then by CV ID. `explain` runs only for scores of 0.5 or higher.
- `human_review` pauses the graph with `interrupt()`. State is saved by `SqliteSaver`, and `search_id` is the `thread_id`. When HR submits the approved IDs, the graph resumes with `Command(resume=...)` and generates 6 questions per approved CV: 2 technical, 2 experience and 2 behavioral.

Waiting searches survive a restart as long as `app.db` and `checkpoints.db` are kept. Review submissions are claimed atomically, so a repeated or concurrent submit returns 409.

## Models

The allowlists live in `src/backend/config.py`. Any model name that is not on the list is rejected at startup, so a typo or an expensive model cannot slip in.

| Chat model | Provider | Notes |
|---|---|---|
| `gemini-3.5-flash-lite` (default) | Gemini API | Paced to `GEMINI_MAX_RPM` (free tier: 10 RPM) |
| `gemini-3.5-flash` … `gemini-3.8-flash` | Gemini API | Same pacing |
| `deepseek-flash` | DeepSeek API | Thinking disabled |
| `gpt-6-luna` | OpenAI API | `reasoning_effort: none` |
| `qwen3.8-27b`, `qwen3.5-2b`, `qwen3.5-4b` | Self-hosted vLLM | No API cost; thinking disabled |

| Embedding model | Dimensions | Chroma collection |
|---|---|---|
| `gemini-embedding-2` (default) | 768 | `cv_gemini_embedding_2_768_v1` |
| `text-embedding-3-small` | 1536 | `cv_text_embedding_3_small_1536_v1` |
| `text-embedding-3-large` | 3072 | `cv_text_embedding_3_large_3072_v1` |

Notes:

- Gemini uses the `google-genai` SDK. DeepSeek, OpenAI and vLLM use the same OpenAI-compatible Chat Completions client.
- Every provider gets the same JSON contract, validated with Pydantic. `qwen3.5-2b` and `qwen3.5-4b` are the only exceptions: they get a plain list of keys instead of a JSON Schema and use Qwen's recommended sampling, because with greedy decoding they loop or echo the schema.
- Each embedding model has its own Chroma collection, so vectors from different models never mix. After switching the embedding model, re-ingest the CVs.
- `MIN_SIMILARITY` (0.55) was set for Gemini embeddings. OpenAI embeddings give much lower cosine scores, so recalibrate the threshold if you switch.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | Chat model from the allowlist |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model from the allowlist |
| `GEMINI_API_KEY` | unset | Needed when a Gemini model is chosen (`GOOGLE_API_KEY` also works) |
| `DEEPSEEK_API_KEY` | unset | Needed for `deepseek-flash` |
| `OPENAI_API_KEY` | unset | Needed for `gpt-6-luna` and OpenAI embeddings |
| `VLLM_BASE_URL` | `http://127.0.0.1:8022/v1` | vLLM endpoint for `qwen3.8-27b` |
| `VLLM_QWEN3_5_2B_URL` / `VLLM_QWEN3_5_4B_URL` | ports `8024` / `8027` | vLLM endpoints for the Qwen3.5 models |
| `VLLM_API_KEY` | unset | Only if the vLLM server uses `--api-key` |
| `CHAT_MAX_OUTPUT_TOKENS` | `1024` | Output cap for OpenAI-compatible providers (256–8192). Hitting it counts as an error |
| `GEMINI_TIMEOUT_SECONDS` | `60` | Timeout per request, for every cloud provider |
| `GEMINI_MAX_RPM` | `10` | Gemini chat pacing (1–10), shared by all processes using the same `app.db` |
| `GEMINI_MAX_RETRIES` | `1` | SDK retries for embeddings |
| `MAX_EXTRACT_ATTEMPTS` | `2` | Extraction attempts when fields are missing |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Local fallback; any Qwen tag |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OLLAMA_KEEP_ALIVE` / `OLLAMA_NUM_CTX` | `30m` / `4096` | Fixed; `KEEP_ALIVE` may also be `-1` |
| `LLM_INPUT_MAX_CHARS` | `8000` | Input cap for chat calls |
| `EMBED_BATCH_SIZE` | `20` | Embedding group size (1–20) |
| `MIN_SIMILARITY` | `0.55` | Retrieval cutoff |
| `DATA_DIR` | `data` | Root for `app.db`, `checkpoints.db`, `chroma/`, `uploads/` |

`CHROMA_COLLECTION` and `EMBEDDING_DIMENSIONS` are derived from the embedding model, so leave them unset. Older `GEMINI_CHAT_MODEL` / `GEMINI_EMBEDDING_MODEL` names are still read. Never commit `.env`.

**Limits:**

- Uploads: 20 files per request, 10 MB per file, 100 PDF pages, 24,000 characters of text.
- Queries: 3,000 characters, `top_k` 1–20.
- Graphs: recursion limit 25, 4 branches at a time.
- Scanned PDFs (OCR), user accounts and automatic hiring decisions are out of scope.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/cvs` | Upload PDFs (multipart field `files`) |
| GET | `/cvs` | List CVs; `status`, `limit` (1–200), `offset` |
| GET | `/cvs/{id}` | CV detail |
| DELETE | `/cvs/{id}` | Delete the CV, its vector and its uploaded file |
| POST | `/searches` | Run a search up to the review step |
| POST | `/searches/{id}/review` | Approve CVs and generate questions; 409 if not waiting |
| GET | `/searches` | Search history |
| GET | `/searches/{id}` | Saved criteria, shortlist, decisions and questions |
| GET | `/health` | `{"status": "ok"}` |

The corpus tab also has `/corpus` endpoints. See [Corpus tools](#corpus-tools).

## Tests

```bash
uv run pytest -q
node --test tests/frontend.test.cjs
```

The tests use real SQLite, LangGraph, Chroma and PDF parsing. Model and embedding calls are replaced with fakes. This means the tests check the logic, branches and failure paths, not model quality; model quality is covered by the benchmark below. CI runs both suites on every push and pull request (`.github/workflows/test.yml`).

## Model benchmark

`scripts/benchmark.py` compares chat and embedding models on a fixed dataset.

**Data** (`data/benchmark/`, see its README):

- **`cvs.jsonl`:** 100 CVs from the Kaggle resume dataset, 10 per category. The categories mix easy ones (CHEF, HEALTHCARE, SALES) with groups that are easy to confuse (FINANCE / ACCOUNTANT / BANKING, DESIGNER / DIGITAL-MEDIA, IT / ENGINEERING).
- **`jds.jsonl`:** 15 written job descriptions, one per category plus 5 harder ones in the confusable groups. Each has reference skills, minimum years and level.
- **`gold_extract.jsonl`:** reference skills and years of experience for 30 CVs.

**Experiments and metrics:**

| Experiment | What it measures |
|---|---|
| `extract` | CV extraction: valid JSON rate, skill precision / recall / F1 (exact and lenient), years MAE, latency, tokens, cost |
| `parse` | JD → criteria: skill F1, `min_years` and `level` match |
| `retrieve` | Search quality with each chat model's extractions: P@5, P@10, MRR (a hit is a CV of the JD's category), before and after `score_cv` reranking |
| `embed` | The same retrieval metrics per embedding model, with extractions held fixed |

**Run:**

```bash
uv run python scripts/benchmark.py extract  --models gemini-3.5-flash-lite deepseek-flash
uv run python scripts/benchmark.py parse    --models gemini-3.5-flash-lite deepseek-flash
uv run python scripts/benchmark.py retrieve --models gemini-3.5-flash-lite deepseek-flash
uv run python scripts/benchmark.py embed    --models text-embedding-3-small text-embedding-3-large
uv run python scripts/benchmark.py report
```

How the script runs:

- **Isolation:** each model runs in its own subprocess, with the app's data paths pointed into `results/benchmark/`, so your real database is never touched.
- **Caching:** every call is cached in `results/benchmark/cache.jsonl`, so a re-run only calls what is missing. Add `--retry-errors` to re-run failed calls.
- **Retries:** transient errors (429, 503, timeouts) are retried with backoff before they are recorded. After 3 connection or rate-limit errors in a row, the run stops.
- **Output:** `report` writes `summary.csv` (one row per experiment and model) and `calls.csv` (one row per call).

**Results:**

| Chat model | Extract skill F1 | Years MAE | Parse skill F1 | Retrieve P@5 | p50 latency | Cost / 100 CVs |
|---|---|---|---|---|---|---|
| deepseek-flash | **0.54** | **4.76** | **0.75** | 0.78 | 1.62 s | $0.062 |
| gpt-6-luna | 0.50 | 5.57 | 0.65 | **0.85** | 2.71 s | $0.023 |
| qwen3.5-4b | 0.47 | 6.77 | 0.60 | 0.84 | 2.10 s | $0 |
| qwen3.8-27b | 0.46 | 6.98 | 0.72 | **0.85** | 3.23 s | $0 |
| gemini-3.5-flash-lite | 0.45 | 7.08 | 0.66 | 0.80 | 1.96 s | $0.091* |
| qwen3.5-2b | 0.23 | 11.2 | 0.44 | 0.70 | 0.86 s | $0 |

\* Paid-tier price. The app runs on the free tier.

| Embedding model | P@5 | P@10 | MRR | Cost / 100 CVs |
|---|---|---|---|---|
| text-embedding-3-large | **0.88** | **0.76** | 1.00 | $0.0019 |
| text-embedding-3-small | 0.87 | 0.71 | 1.00 | $0.0002 |
| gemini-embedding-2 | 0.80 | 0.71 | 0.97 | free tier |

What the results show:

- The embedding model affects retrieval more than the chat model does.
- Reranking with `score_cv` lowers P@k for every model, so the weights need tuning.
- Years of experience is the weakest field for every model.

Full numbers are in `results/benchmark/summary.csv`.

## Corpus tools

The **Corpus** tab works with the Djinni candidate-profile dataset. It needs `uv sync --extra research` for the Parquet reader.

- **Create sample:** builds a JSONL sample from `data/datasets/djinni/train.parquet`.
- **Verify:** checks a sample against its manifest checksum.
- **Extract + embed:** ingests 1–50 profiles per run. A cursor remembers where the last run stopped.
- **Benchmark:** runs a local SQLite FTS5/BM25 keyword baseline. It makes no model calls.

Only one corpus job runs at a time, and jobs survive server restarts. Contact details are masked before extraction, but this is not full anonymization. See [docs/DATASET.md](docs/DATASET.md) for the data source, row counts, checksums and sampling.

## Other scripts

| Command | Purpose |
|---|---|
| `uv run python -m scripts.check_gemini_model` | One short Gemini request to check the key and network; sends no CVs |
| `uv run python -m scripts.diagnose_gemini_connection` | Compares Python and curl HTTPS access when the check above times out |
| `uv run python -m scripts.export_graphs` | Exports the ingest and search graph diagrams |
| `.\scripts\run-real-data.ps1 -Limit 20` | Windows pilot run: checks, tests, then ingests 20 corpus CVs |

## Further reading

- [docs/INGEST_RULES.md](docs/INGEST_RULES.md): ingest rules and the two-phase recovery contract.
- [labs/state_checkpoint](labs/state_checkpoint): a small hands-on lab on state, reducers and checkpoints (no API key needed).
