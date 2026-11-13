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
        settings.CHAT_MODEL = "gemini-3.5-flash-lite"
        settings.CHAT_PROVIDER = "gemini"
        settings.CLOUD_CHAT_MODELS = {"gemini-3.5-flash-lite": {"provider": "gemini", "price": (0.30, 2.50)},
                                      "deepseek-flash": {"provider": "deepseek", "price": (0.30, 1.20),
                                                         "extra_body": {"thinking": {"type": "disabled"}}},
                                      "gpt-6-luna": {"provider": "openai", "price": (0.10, 0.50),
                                                     "extra_body": {"reasoning_effort": "none"}}}
        settings.CLOUD_CHAT_MODELS["gemini-3.5-flash-lite"]["thinking_level"] = "MINIMAL"
        settings.CLOUD_CHAT_MODELS["qwen3.8-27b"] = {
            "provider": "vllm", "price": (0.0, 0.0),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        settings.OPENAI_COMPAT_PROVIDERS = {
            "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", True),
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", True),
            "vllm": ("http://127.0.0.1:8022/v1", "VLLM_API_KEY", False)}
        settings.OLLAMA_CHAT_MODEL = "qwen2.5:3b"
        settings.EMBEDDING_MODEL = "gemini-embedding-2"
        settings.EMBEDDING_PROVIDER = "gemini"
        settings.EMBEDDING_DIMENSIONS = 768
        settings.CHAT_RPM = 12
        settings.GEMINI_TIMEOUT_SECONDS = 30
        settings.GEMINI_MAX_RETRIES = 1
        settings.GEMINI_MAX_RPM = 10
        settings.TEMPERATURE = 0.0
        settings.CHAT_MAX_OUTPUT_TOKENS = 1024
        settings.OLLAMA_BASE_URL = "http://ollama.test:11434"
        settings.OLLAMA_KEEP_ALIVE = "30m"
        settings.OLLAMA_NUM_CTX = 4096
        settings.OLLAMA_TIMEOUT_SECONDS = 180
        settings.EMBED_TEXT_MAX_WORDS = 300
        settings.EMBED_BATCH_SIZE = 20
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
        self.services.gemini_pacer = Mock(return_value=Mock())
        self.settings = settings

    def sdk_modules(self):
        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = Mock()
        sdk_types = types.ModuleType("google.genai.types")
        sdk_types.HttpOptions = lambda **kwargs: kwargs
        sdk_types.HttpRetryOptions = lambda **kwargs: kwargs
        sdk_types.GenerateContentConfig = lambda **kwargs: kwargs
        sdk_types.ThinkingConfig = lambda **kwargs: kwargs
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
        self.assertEqual(kwargs["config"]["thinking_config"], {"thinking_level": "MINIMAL"})

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

    def use_openai_compat(self, model, provider):
        self.settings.CHAT_MODEL, self.settings.CHAT_PROVIDER = model, provider

    def _http_response(self, content):
        body = json.dumps({"choices": [{"message": {"content": content}}],
                           "usage": {"prompt_tokens": 12, "completion_tokens": 4}}).encode()
        response = Mock()
        response.read.return_value = body
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_openai_compatible_providers_send_json_mode_to_their_endpoint(self):
        for model, provider, env, url, no_thinking in (
                ("deepseek-flash", "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions",
                 {"thinking": {"type": "disabled"}}),
                ("gpt-6-luna", "openai", "OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions",
                 {"reasoning_effort": "none"})):
            self.use_openai_compat(model, provider)
            urlopen = Mock(return_value=self._http_response('{"skills": ["Python"]}'))
            with self.subTest(model=model), patch.dict(os.environ, {env: "test-key"}, clear=True), \
                    patch("urllib.request.urlopen", urlopen):
                out = self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Python"})
            self.assertEqual(out.skills, ["Python"])
            request = urlopen.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(request.full_url, url)
            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
            self.assertEqual(body["model"], model)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(body["temperature"], 0.0)
            self.assertIn("JSON Schema", body["messages"][0]["content"])
            for key, value in no_thinking.items():
                self.assertEqual(body[key], value)
            cap = "max_completion_tokens" if provider == "openai" else "max_tokens"
            self.assertEqual(body[cap], 1024)

    def test_vllm_needs_no_key_and_disables_qwen_thinking(self):
        self.use_openai_compat("qwen3.8-27b", "vllm")
        urlopen = Mock(return_value=self._http_response('{"skills": ["Go"]}'))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", urlopen):
            out = self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Go"})
        self.assertEqual(out.skills, ["Go"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8022/v1/chat/completions")
        self.assertIsNone(request.get_header("Authorization"))
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen3.8-27b")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_last_call_stats_records_pacing_wait_and_tokens(self):
        self.services.gemini_pacer = Mock(return_value=Mock(acquire=Mock(return_value=2.5)))
        _, modules = self.sdk_modules()
        client = self.gemini_client(text='{"skills": ["Python"]}')
        with patch.dict(sys.modules, modules), \
                patch.object(self.services, "chat_client", return_value=client):
            self.services.cloud_json(self.services.SearchCriteria, "Extract criteria", {"query": "Python"})
        self.assertEqual(self.services.last_call_stats.get(), {
            "model": "gemini-3.5-flash-lite", "wait_s": 2.5, "input_tokens": 10, "output_tokens": 5})
        self.use_openai_compat("qwen3.8-27b", "vllm")
        urlopen = Mock(return_value=self._http_response('not json'))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", urlopen):
            with self.assertRaises(ValueError):
                self.services.cloud_json(self.services.SearchCriteria, "Extract criteria", {"query": "Go"})
        # Usage is still recorded when the reply fails to parse.
        self.assertEqual(self.services.last_call_stats.get(), {
            "model": "qwen3.8-27b", "wait_s": 2.5, "input_tokens": 12, "output_tokens": 4})

    def test_output_hitting_the_token_cap_is_an_error(self):
        self.use_openai_compat("qwen3.8-27b", "vllm")
        body = json.dumps({"choices": [{"finish_reason": "length",
                                        "message": {"content": '{"skills": ["Go", "Go", "Go"'}}]}).encode()
        response = self._http_response("unused")
        response.read.return_value = body
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", Mock(return_value=response)):
            with self.assertRaisesRegex(ValueError, "token cap"):
                self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Go"})

    def test_model_specific_base_url_overrides_the_provider_url(self):
        self.settings.CLOUD_CHAT_MODELS["qwen3.5-2b"] = {
            "provider": "vllm", "price": (0.0, 0.0), "base_url": "http://127.0.0.1:8024/v1/",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        self.use_openai_compat("qwen3.5-2b", "vllm")
        urlopen = Mock(return_value=self._http_response('{"skills": ["Go"]}'))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", urlopen):
            self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Go"})
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://127.0.0.1:8024/v1/chat/completions")

    def test_key_list_prompt_and_sampling_override_for_small_models(self):
        self.settings.CLOUD_CHAT_MODELS["qwen3.5-4b"] = {
            "provider": "vllm", "price": (0.0, 0.0), "prompt": "keys",
            "extra_body": {"temperature": 0.7, "presence_penalty": 1.5}}
        self.use_openai_compat("qwen3.5-4b", "vllm")
        urlopen = Mock(return_value=self._http_response('{"skills": ["Go"]}'))
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", urlopen):
            self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Go"})
        body = json.loads(urlopen.call_args.args[0].data)
        system = body["messages"][0]["content"]
        self.assertIn("skills (array of strings)", system)
        self.assertNotIn("JSON Schema", system)
        self.assertEqual((body["temperature"], body["presence_penalty"]), (0.7, 1.5))

    def test_require_models_checks_keys_of_configured_providers(self):
        self.use_openai_compat("deepseek-flash", "deepseek")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g"}, clear=True):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.require_models()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g", "DEEPSEEK_API_KEY": "d"}, clear=True):
            self.services.require_models()
        self.settings.EMBEDDING_PROVIDER, self.settings.EMBEDDING_MODEL = "openai", "text-embedding-3-small"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g", "DEEPSEEK_API_KEY": "d"}, clear=True):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.require_models()
        self.use_openai_compat("qwen3.8-27b", "vllm")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "o"}, clear=True):
            self.services.require_models()  # no Gemini key needed: neither model is Gemini

    def test_unpaced_models_skip_the_shared_pacer(self):
        self.settings.CHAT_RPM = None
        with patch.object(self.services, "_sleep") as sleep:
            self.assertEqual(self.services.SharedPacingLimiter().acquire(), 0.0)
        sleep.assert_not_called()

    def test_openai_compatible_provider_requires_its_own_key(self):
        self.use_openai_compat("deepseek-flash", "deepseek")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with self.assertRaises(self.services.ConfigurationError):
                self.services.structured(self.services.SearchCriteria, "Extract criteria", {"query": "Python"})

    def test_openai_compatible_429_and_timeout_reroute_to_ollama(self):
        import urllib.error
        self.use_openai_compat("gpt-6-luna", "openai")
        record = json.dumps({"name": "A", "role": "R", "skills": ["Go"], "exp_yrs": 2, "edu": "", "summary": ""})
        for error in (urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None),
                      urllib.error.URLError(TimeoutError("timed out")), TimeoutError("read")):
            post = Mock(return_value=self._ollama_response(record))
            with self.subTest(error=repr(error)), patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True), \
                    patch("urllib.request.urlopen", Mock(side_effect=error)), \
                    patch.object(self.services, "_ollama_post", post):
                _, source = self.services.extract_cv("Go CV", [])
            self.assertEqual(source, "ollama")

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


    def test_query_embedding_429_backoff_never_calls_local_or_retries_chroma(self):
        embedding = Mock()
        embedding.embed_query.side_effect = [FakeAPIError(429), FakeAPIError(429), [1., 0.]]
        store = Mock()
        store.count.return_value = 1
        store.query.return_value = {"ids": [["cv"]], "distances": [[0.]]}
        self.settings.MIN_SIMILARITY = .55
        with patch.object(self.services, "embeddings", return_value=embedding), \
                patch.object(self.services, "collection", return_value=store), \
                patch.object(self.services, "_sleep") as sleep, \
                patch.object(self.services.random, "uniform", return_value=2.), \
                patch.object(self.services, "_ollama_post") as local:
            self.assertEqual(self.services.vector_search("Python", 1), [{"cv_id": "cv", "similarity": 1.}])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [12., 22.])
        self.assertEqual(embedding.embed_query.call_count, 3)
        store.query.assert_called_once()
        local.assert_not_called()

    def test_ollama_rejects_non_object_and_error_responses_before_reporting_ready(self):
        for payload in (None, [], {"error": "model missing"}):
            with self.subTest(payload=payload), \
                    patch.object(self.services.urllib.request, "urlopen") as request:
                request.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
                self.assertFalse(self.services.ollama_warmup())
