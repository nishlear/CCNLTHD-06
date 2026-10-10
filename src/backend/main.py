from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import db
from .routers import cvs, searches

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="JD Matching", lifespan=lifespan)
app.include_router(cvs.router)
app.include_router(searches.router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Mounted last so API routes win. Serves index.html at "/".
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
