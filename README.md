# JD Matching

HR uploads CV PDFs, searches them with free text, reviews a shortlist, and gets interview questions for approved candidates. The workflow is built with LangGraph. Design: `docs/personal/design-2.0.md`.

> The LangGraph code is not in yet. `src/backend/graphs/` holds "change me" stubs so the API and UI work end to end.

## Run

### Docker (recommended)

Needs Docker + Docker Compose.

```bash
docker compose up -d --build
```

Open http://localhost:8000. Stop with `docker compose down`.

Data (SQLite + uploaded PDFs) is saved in `./data`. If a `.env` file exists, it is loaded into the container.

### Local

Needs [uv](https://docs.astral.sh/uv/). Python 3.11 is installed by uv if missing.

```bash
uv sync
uv run uvicorn src.backend.main:app --reload
```

Open http://localhost:8000. API docs are at http://localhost:8000/docs.

## Project structure

```
src/
  backend/
    main.py          FastAPI app, also serves the frontend
    routers/         API endpoints (/cvs, /searches)
    graphs/          LangGraph code goes here (stubs for now)
      ingest.py      run_ingest()                     — CV ingest graph
      search.py      start_search(), resume_search()  — search graph
    schemas.py       Pydantic models (shared with the graphs)
    db.py            SQLite helpers (app.db)
    config.py        Paths and limits
  frontend/          Plain HTML/JS/CSS, no build step
```

## Plugging in the real graphs

Only `src/backend/graphs/` needs to change. Each stub's docstring says what it must return.

- `run_ingest(files)` → list of `{cv_id, file_name, status, error}`. The graph saves the CV row itself (`db.save_cv`).
- `start_search(id, query, top_k)` → `{paused, criteria, shortlist, explanations}`. Use `id` as the LangGraph `thread_id`.
- `resume_search(id, approved_cv_ids)` → list of `{cv_id, category, question}`.

The API layer writes search results and questions to `app.db`, so the graph nodes don't need to.

## Config

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Where SQLite files and uploads go |
| `APP_DB_PATH` | `$DATA_DIR/app.db` | App database |
| `CHECKPOINT_DB_PATH` | `$DATA_DIR/checkpoints.db` | LangGraph checkpoints |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/cvs` | Upload PDFs (multipart, field `files`) |
| GET | `/cvs` | List CVs, optional `?status=ready\|failed` |
| GET | `/cvs/{id}` | CV detail |
| DELETE | `/cvs/{id}` | Delete a CV |
| POST | `/searches` | Start a search, returns shortlist (`awaiting_review`) |
| POST | `/searches/{id}/review` | Send approved ids, returns questions (`409` if not awaiting review) |
| GET | `/searches` | List searches |
| GET | `/searches/{id}` | Search detail |
| GET | `/health` | `{"status": "ok"}` |
