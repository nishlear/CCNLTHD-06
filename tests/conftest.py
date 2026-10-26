import pytest
from fastapi.testclient import TestClient

from src.backend import config, db, services
from src.backend.main import app


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(db, "APP_DB_PATH", tmp_path / "app.db")
    config.UPLOAD_DIR.mkdir()
    services.collection.cache_clear()
    services.chat_client.cache_clear()
    services.gemini_pacer.cache_clear()
    services.embeddings.cache_clear()
    db.init_db()
    yield tmp_path
    services.collection.cache_clear()


@pytest.fixture
def model_doubles(isolated, monkeypatch):
    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[1., 0., 0.] for _ in texts]
    monkeypatch.setattr(services, "embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(services, "require_gemini", lambda: None)
    monkeypatch.setattr(services, "parse_query", lambda query: {
        "skills": ["Python", "FastAPI"], "min_years": 3})
    monkeypatch.setattr(services, "explain_match", lambda cv, criteria, result: "Fixture explanation")
    monkeypatch.setattr(services, "generate_questions", lambda cv_id, cv, criteria: [
        {"cv_id": cv_id, "category": category, "question": f"Fixture {category}?"}
        for category in ("technical", "experience", "behavioral")])


@pytest.fixture
def candidates(model_doubles, monkeypatch):
    db.save_cv("cv-a", "a.pdf", "ready", extracted={
        "full_name": "Candidate A", "skills": ["Python", "FastAPI"], "years_experience": 4})
    db.save_cv("cv-b", "b.pdf", "ready", extracted={
        "full_name": "Candidate B", "skills": ["Python"], "years_experience": None})
    monkeypatch.setattr(services, "vector_search", lambda query, k: [
        {"cv_id": "cv-a", "similarity": 1}, {"cv_id": "cv-b", "similarity": .9}])


@pytest.fixture
def client(model_doubles):
    with TestClient(app) as session:
        yield session
