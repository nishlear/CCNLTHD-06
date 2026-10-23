"""Provider boundary tests; clients are doubles and no real API is called."""

import importlib.util
import os
import sys
import types
import unittest
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import Mock, patch


class GeminiServiceTests(unittest.TestCase):
    def setUp(self):
        settings = types.ModuleType("src.backend.config")
        settings.DEFAULT_TOP_K, settings.MAX_TOP_K = 10, 20
        settings.GEMINI_CHAT_MODEL = "gemini-3.8-flash"
        settings.GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
        settings.GEMINI_EMBEDDING_DIMENSIONS = 768
        settings.GEMINI_TIMEOUT_SECONDS = 30
        settings.GEMINI_MAX_RETRIES = 1
        settings.CHROMA_DIR = Path("data/test-gemini-unused")
        settings.CHROMA_COLLECTION = "fixture-gemini-collection"
        tracing = types.ModuleType("src.backend.tracing")
        tracing.thread_id = ContextVar("test-thread", default=None)
        spec = importlib.util.spec_from_file_location("src.backend._gemini_services_test",
            Path(__file__).resolve().parents[1] / "src/backend/services.py")
        self.services = importlib.util.module_from_spec(spec)
        import src.backend as backend
        with patch.dict(sys.modules, {"src.backend.config": settings, "src.backend.tracing": tracing}), \
                patch.object(backend, "config", settings, create=True):
            spec.loader.exec_module(self.services)
        self.settings = settings

    def test_openai_key_is_not_accepted_and_gemini_alias_is_supported(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "old-provider-key"}, clear=True):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.require_gemini()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-test-key"}, clear=True):
            self.services.require_gemini()
            self.assertEqual(self.services.gemini_api_key(), "google-test-key")

    def test_chat_client_explicitly_uses_gemini_developer_api(self):
        package = types.ModuleType("langchain_google_genai")
        package.ChatGoogleGenerativeAI = Mock()
        with patch.dict(sys.modules, {"langchain_google_genai": package}), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key", "GOOGLE_API_KEY": "other-key"}, clear=True):
            self.services.chat_model()
        kwargs = package.ChatGoogleGenerativeAI.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "gemini-test-key")
        self.assertFalse(kwargs["vertexai"])
        self.assertEqual(kwargs["timeout"], 30)
        self.assertEqual(kwargs["max_retries"], 1)
        self.assertNotIn("temperature", kwargs)

    def test_embedding_sdk_timeout_units_and_retry_attempts(self):
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = Mock()
        sdk_types = types.ModuleType("google.genai.types")
        sdk_types.HttpOptions = lambda **kwargs: kwargs
        sdk_types.HttpRetryOptions = lambda **kwargs: kwargs
        google.genai, genai.types = genai, sdk_types
        with patch.dict(sys.modules, {"google": google, "google.genai": genai, "google.genai.types": sdk_types}), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key"}, clear=True):
            self.services.embeddings()
        kwargs = genai.Client.call_args.kwargs
        self.assertFalse(kwargs["vertexai"])
        self.assertEqual(kwargs["http_options"]["timeout"], 30000)
        self.assertEqual(kwargs["http_options"]["retry_options"]["attempts"], 2)

    def test_native_structured_output_is_validated(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": {"skills": ["Python"]}, "raw": None, "parsing_error": None}
        with patch.object(self.services, "chat_model", return_value=model):
            out = self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Python"})
        self.assertEqual(out.skills, ["Python"])
        self.assertEqual(model.with_structured_output.call_args.kwargs,
                         {"method": "json_schema", "include_raw": True})

    def test_legacy_collection_is_rejected_and_index_uses_document_embedding(self):
        chroma = types.ModuleType("chromadb")
        chroma_config = types.ModuleType("chromadb.config")
        chroma_config.Settings = Mock()
        store = types.SimpleNamespace(metadata={"embedding_model": "text-embedding-3-small"})
        chroma.PersistentClient = Mock(return_value=types.SimpleNamespace(
            get_or_create_collection=Mock(return_value=store)))
        with patch.dict(sys.modules, {"chromadb": chroma, "chromadb.config": chroma_config}):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.collection()
        embedding = Mock()
        embedding.embed_documents.return_value = [[1., 0., 0.]]
        store = Mock()
        with patch.object(self.services, "embeddings", return_value=embedding), \
                patch.object(self.services, "collection", return_value=store):
            self.services.index_cv("cv", {"skills": ["Python"]}, "Python CV")
        embedding.embed_query.assert_not_called()
        store.upsert.assert_called_once_with(ids=["cv"], embeddings=[[1., 0., 0.]])
