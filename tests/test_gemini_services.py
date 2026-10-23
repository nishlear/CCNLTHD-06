"""Provider boundary tests; clients are doubles and no real API is called."""

import importlib.util
import json
import os
import sys
import types
import unittest
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import Mock, patch


class FakeAPIError(Exception):
    def __init__(self, code, status=""):
        super().__init__(f"{code} {status}")
        self.code, self.status = code, status


class TimeoutException(Exception):
    """Same class name as httpx.TimeoutException, which is not a TimeoutError."""


class GeminiServiceTests(unittest.TestCase):
    def setUp(self):
        settings = types.ModuleType("src.backend.config")
        settings.DEFAULT_TOP_K, settings.MAX_TOP_K = 10, 20
        settings.GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
        settings.OLLAMA_CHAT_MODEL = "qwen2.5:3b"
        settings.GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
        settings.GEMINI_EMBEDDING_DIMENSIONS = 768
        settings.GEMINI_TIMEOUT_SECONDS = 30
        settings.GEMINI_MAX_RETRIES = 1
        settings.GEMINI_MAX_RPM = 12
        settings.TEMPERATURE = 0.0
        settings.OLLAMA_BASE_URL = "http://ollama.test:11434"
        settings.OLLAMA_KEEP_ALIVE = "30m"
        settings.OLLAMA_NUM_CTX = 4096
        settings.OLLAMA_TIMEOUT_SECONDS = 180
        settings.EMBED_TEXT_MAX_WORDS = 300
        settings.EMBED_BACKOFF_SECONDS = (10, 20, 40)
        settings.EMBED_BACKOFF_JITTER = (1.0, 3.0)
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

    def sdk_modules(self):
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = Mock()
        sdk_types = types.ModuleType("google.genai.types")
        sdk_types.HttpOptions = lambda **kwargs: kwargs
        sdk_types.HttpRetryOptions = lambda **kwargs: kwargs
        sdk_types.GenerateContentConfig = lambda **kwargs: kwargs
        google.genai, genai.types = genai, sdk_types
        return genai, {"google": google, "google.genai": genai, "google.genai.types": sdk_types}

    def gemini_client(self, text=None, error=None):
        client = Mock()
        if error is not None:
            client.models.generate_content.side_effect = error
        else:
            client.models.generate_content.return_value = types.SimpleNamespace(
                text=text, usage_metadata=types.SimpleNamespace(prompt_token_count=10, candidates_token_count=5))
        return client

    def test_openai_key_is_not_accepted_and_gemini_alias_is_supported(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "old-provider-key"}, clear=True):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.require_gemini()
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-test-key"}, clear=True):
            self.services.require_gemini()
            self.assertEqual(self.services.gemini_api_key(), "google-test-key")

    def test_chat_client_uses_developer_api_without_sdk_retries(self):
        genai, modules = self.sdk_modules()
        with patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key", "GOOGLE_API_KEY": "other-key"}, clear=True):
            self.services.chat_client()
        kwargs = genai.Client.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "gemini-test-key")
        self.assertFalse(kwargs["vertexai"])
        self.assertEqual(kwargs["http_options"]["timeout"], 30000)
        # Rule 3.2: a 429 must reroute immediately, never be retried by the SDK.
        self.assertEqual(kwargs["http_options"]["retry_options"], {"attempts": 1})

    def test_embedding_sdk_timeout_units_and_retry_excludes_429(self):
        genai, modules = self.sdk_modules()
        with patch.dict(sys.modules, modules), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test-key"}, clear=True):
            self.services.embeddings()
        kwargs = genai.Client.call_args.kwargs
        self.assertFalse(kwargs["vertexai"])
        self.assertEqual(kwargs["http_options"]["timeout"], 30000)
        retry = kwargs["http_options"]["retry_options"]
        self.assertEqual(retry["attempts"], 2)
        self.assertNotIn(429, retry["http_status_codes"])

    def test_gemini_call_forces_json_mime_schema_and_zero_temperature(self):
        _, modules = self.sdk_modules()
        client = self.gemini_client(text='{"skills": ["Python"]}')
        with patch.dict(sys.modules, modules), \
                patch.object(self.services, "chat_client", return_value=client), \
                patch.object(self.services, "_sleep"):
            out = self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Python"})
        self.assertEqual(out.skills, ["Python"])
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")
        self.assertEqual(kwargs["config"]["temperature"], 0.0)
        self.assertIn("properties", kwargs["config"]["response_json_schema"])
        self.assertIn("JSON object", kwargs["config"]["system_instruction"])

    def test_pacer_spaces_calls_five_seconds_apart_at_12_rpm(self):
        clock = Mock(return_value=100.0)
        sleeps = []
        pacer = self.services.PacingLimiter(12, clock=clock, sleep=sleeps.append)
        for _ in range(3):
            pacer.acquire()
        self.assertEqual(sleeps, [5.0, 10.0])
        clock.return_value = 200.0
        pacer.acquire()
        self.assertEqual(sleeps, [5.0, 10.0])

    def _ollama_response(self, content):
        return {"message": {"content": content}, "prompt_eval_count": 50, "eval_count": 20}

    def test_extract_reroutes_to_ollama_on_429_and_timeout_without_retrying_gemini(self):
        _, modules = self.sdk_modules()
        record = json.dumps({"name": "A", "role": "Backend", "skills": "Python, FastAPI",
                             "exp_yrs": "5+ years", "edu": None, "summary": "Builds APIs"})
        for error in (FakeAPIError(429, "RESOURCE_EXHAUSTED"), TimeoutException("read"),
                      FakeAPIError(504, "DEADLINE_EXCEEDED"), FakeAPIError(503, "UNAVAILABLE")):
            client = self.gemini_client(error=error)
            post = Mock(return_value=self._ollama_response(record))
            with self.subTest(error=repr(error)), patch.dict(sys.modules, modules), \
                    patch.object(self.services, "chat_client", return_value=client), \
                    patch.object(self.services, "_ollama_post", post), \
                    patch.object(self.services, "_sleep"):
                data, source = self.services.extract_cv("Python CV", [])
            self.assertEqual(source, "ollama")
            self.assertEqual(client.models.generate_content.call_count, 1)
            self.assertEqual(data, {"full_name": "A", "role": "Backend", "skills": ["Python", "FastAPI"],
                                    "years_experience": 5.0, "education": None, "summary": "Builds APIs"})
            path, body, _ = post.call_args.args
            self.assertEqual(path, "/api/chat")
            self.assertEqual(body["model"], "qwen2.5:3b")
            self.assertEqual(body["format"], "json")
            self.assertEqual(body["keep_alive"], "30m")
            self.assertEqual(body["options"], {"temperature": 0.0, "num_ctx": 4096})
            self.assertFalse(body["stream"])

    def test_both_providers_receive_the_same_prompt(self):
        _, modules = self.sdk_modules()
        ok = json.dumps({"name": "A", "role": "R", "skills": ["Go"], "exp_yrs": 2, "edu": "", "summary": ""})
        client = self.gemini_client(text=ok)
        post = Mock(return_value=self._ollama_response(ok))
        with patch.dict(sys.modules, modules), \
                patch.object(self.services, "chat_client", return_value=client), \
                patch.object(self.services, "_ollama_post", post), patch.object(self.services, "_sleep"):
            gemini_data, source = self.services.extract_cv("Go CV", ["skills"])
            ollama_data = self.services.ollama_json(self.services.CVExtract, self.services.EXTRACT_INSTRUCTION,
                                                    {"cv": "Go CV", "fix": ["skills"]}).to_record()
        self.assertEqual(source, "gemini")
        self.assertEqual(gemini_data, ollama_data)
        gemini_call = client.models.generate_content.call_args.kwargs
        messages = post.call_args.args[1]["messages"]
        self.assertEqual(messages[0]["content"], gemini_call["config"]["system_instruction"])
        self.assertEqual(messages[1]["content"], gemini_call["contents"])

    def test_non_capacity_gemini_errors_do_not_fail_over(self):
        _, modules = self.sdk_modules()
        client = self.gemini_client(error=FakeAPIError(400, "INVALID_ARGUMENT"))
        post = Mock()
        with patch.dict(sys.modules, modules), \
                patch.object(self.services, "chat_client", return_value=client), \
                patch.object(self.services, "_ollama_post", post), patch.object(self.services, "_sleep"):
            with self.assertRaises(FakeAPIError):
                self.services.extract_cv("CV", [])
        post.assert_not_called()

    def test_ollama_keep_alive_minus_one_is_sent_as_integer(self):
        self.settings.OLLAMA_KEEP_ALIVE = "-1"
        self.assertEqual(self.services._ollama_keep_alive(), -1)

    def test_compact_embedding_text_uses_extracted_fields_only(self):
        text = self.services.embedding_text({"role": "Backend Engineer", "skills": ["Python", "FastAPI"],
                                             "years_experience": 4.0, "summary": "Builds APIs"})
        self.assertEqual(text, "Backend Engineer | Skills: Python, FastAPI | Experience: 4 years | Builds APIs")
        long = self.services.embedding_text({"skills": ["x"] * 1000})
        self.assertLessEqual(len(long.split()), 300)
        self.assertTrue(long.startswith("Unknown role | Skills:"))
        self.assertIn("Experience: unknown", self.services.embedding_text({}))

    def test_embedding_429_uses_exponential_backoff_with_jitter(self):
        embedding = Mock()
        embedding.embed_documents.side_effect = [FakeAPIError(429), FakeAPIError(429), [[1., 0.]]]
        sleeps = []
        with patch.object(self.services, "embeddings", return_value=embedding), \
                patch.object(self.services, "_sleep", sleeps.append), \
                patch.object(self.services.random, "uniform", return_value=2.0):
            self.assertEqual(self.services.embed_document_with_backoff("text"), [1., 0.])
        self.assertEqual(sleeps, [12.0, 22.0])
        embedding.embed_documents.side_effect = FakeAPIError(429)
        sleeps.clear()
        with patch.object(self.services, "embeddings", return_value=embedding), \
                patch.object(self.services, "_sleep", sleeps.append), \
                patch.object(self.services.random, "uniform", return_value=1.0):
            with self.assertRaises(FakeAPIError):
                self.services.embed_document_with_backoff("text")
        self.assertEqual(sleeps, [11.0, 21.0, 41.0])
        embedding.embed_documents.side_effect = FakeAPIError(400)
        with patch.object(self.services, "embeddings", return_value=embedding), \
                patch.object(self.services, "_sleep") as sleep:
            with self.assertRaises(FakeAPIError):
                self.services.embed_document_with_backoff("text")
        sleep.assert_not_called()

    def test_legacy_collection_is_rejected_and_index_uses_compact_document_embedding(self):
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
            self.services.index_cv("cv", {"role": "Dev", "skills": ["Python"]})
        embedding.embed_query.assert_not_called()
        embedding.embed_documents.assert_called_once_with(["Dev | Skills: Python | Experience: unknown |"])
        store.upsert.assert_called_once_with(ids=["cv"], embeddings=[[1., 0., 0.]],
                                             metadatas=[{"doc_format": "compact-v2"}])
