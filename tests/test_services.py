import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.backend import config, db, services


def pdf_bytes(text="Candidate A Python FastAPI 4 years"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FixtureEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_real_pdf_extraction_and_blank_pdf_error(isolated):
    path = isolated / "cv.pdf"
    path.write_bytes(pdf_bytes())
    assert "Python FastAPI" in services.pdf_to_text(str(path))
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(str(path))
    with pytest.raises(ValueError, match="OCR"):
        services.pdf_to_text(str(path))


def test_encrypted_pdf_is_rejected(isolated):
    path = isolated / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    writer.write(str(path))
    with pytest.raises(ValueError, match="mã hóa"):
        services.pdf_to_text(str(path))


def test_real_chroma_cosine_normalization(isolated, monkeypatch):
    monkeypatch.setattr(services, "embeddings", lambda: FixtureEmbeddings())
    store = services.collection()
    store.upsert(ids=["a", "b", "c"], embeddings=[[1., 0., 0.], [.8, .6, 0.], [0., 1., 0.]])
    results = services.vector_search("Python", 20)
    assert [r["cv_id"] for r in results] == ["a", "b"]
    assert [r["similarity"] for r in results] == pytest.approx([1, .9])
    services.delete_vector("a")
    assert store.get(ids=["a"])["ids"] == []


def test_empty_collection_does_not_call_embedding(isolated, monkeypatch):
    def unexpected():
        raise AssertionError("Empty collection should not call embedding API")
    monkeypatch.setattr(services, "embeddings", unexpected)
    assert services.vector_search("Python", 20) == []


def test_embedding_model_change_requires_new_collection(isolated, monkeypatch):
    services.collection()
    services.collection.cache_clear()
    monkeypatch.setattr(config, "GEMINI_EMBEDDING_MODEL", "different-model")
    with pytest.raises(services.ConfigurationError):
        services.collection()


def test_pdf_to_review_api_with_real_vector_store(client, monkeypatch):
    monkeypatch.setattr(services, "embeddings", lambda: FixtureEmbeddings())
    monkeypatch.setattr(services, "extract_cv", lambda text, missing: {
        "full_name": "Candidate A", "skills": ["Python", "FastAPI"], "years_experience": 4})
    response = client.post("/cvs", files=[("files", ("cv.pdf", pdf_bytes(), "application/pdf")),
                                         ("files", ("broken.pdf", b"not a PDF", "application/pdf"))])
    assert response.status_code == 200
    rows = response.json()["cvs"]
    assert [r["status"] for r in rows] == ["ready", "failed"]
    cv_id = rows[0]["id"]
    assert db.get_cv(cv_id)["raw_text"]
    result = client.post("/searches", json={"query": "Python FastAPI", "top_k": 2})
    assert result.status_code == 200
    assert result.json()["shortlist"][0]["cv_id"] == cv_id
    search_id = result.json()["id"]
    review = client.post(f"/searches/{search_id}/review", json={"approved_cv_ids": [cv_id]})
    assert review.status_code == 200 and len(review.json()["questions"]) == 3
    assert client.delete(f"/cvs/{cv_id}").status_code == 204
    assert services.collection().get(ids=[cv_id])["ids"] == []
    assert client.get(f"/cvs/{cv_id}").status_code == 404
    assert not (config.UPLOAD_DIR / f"{cv_id}.pdf").exists()


def test_upload_size_limit_has_no_orphans(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 16)
    response = client.post("/cvs", files={"files": ("big.pdf", b"x" * 17, "application/pdf")})
    assert response.status_code == 413
    assert list(config.UPLOAD_DIR.iterdir()) == []
