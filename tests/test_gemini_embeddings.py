import math
import types
import unittest
from unittest.mock import Mock

from src.backend.gemini_embeddings import GeminiEmbeddings


class GeminiEmbeddingsTests(unittest.TestCase):
    def client(self, values=None):
        values = [3., 4., 0.] if values is None else values
        return types.SimpleNamespace(models=types.SimpleNamespace(embed_content=Mock(
            return_value=types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=values)]))))

    def test_embedding_2_uses_asymmetric_prefixes_and_never_legacy_task_type(self):
        client = self.client()
        model = GeminiEmbeddings(client, "gemini-embedding-2", 3)
        self.assertEqual(model.embed_documents(["Python CV"]), [[.6, .8, 0.]])
        first = client.models.embed_content.call_args.kwargs
        self.assertEqual(first["contents"][0].parts[0].text, "title: none | text: Python CV")
        self.assertNotIn("task_type", first["config"])
        self.assertEqual(model.embed_query("Python job"), [.6, .8, 0.])
        self.assertEqual(client.models.embed_content.call_args.kwargs["contents"], "task: search result | query: Python job")

    def test_other_embedding_models_are_rejected(self):
        with self.assertRaises(ValueError):
            GeminiEmbeddings(self.client(), "gemini-embedding-001", 3)

    def test_multiple_documents_are_not_aggregated_into_one_vector(self):
        client = self.client()
        model = GeminiEmbeddings(client, "gemini-embedding-2", 3)
        client.models.embed_content.return_value.embeddings *= 2
        self.assertEqual(len(model.embed_documents(["first", "second"])), 2)
        self.assertEqual(client.models.embed_content.call_count, 1)
        contents = client.models.embed_content.call_args.kwargs["contents"]
        self.assertEqual([item.parts[0].text for item in contents],
                         ["title: none | text: first", "title: none | text: second"])
        client.models.embed_content.reset_mock()
        self.assertEqual(model.embed_documents([]), [])
        client.models.embed_content.assert_not_called()

    def test_invalid_vectors_fail_before_vector_store_upsert(self):
        for values in ([1, 2], [float("nan"), 1, 0], [0, 0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GeminiEmbeddings(self.client(values), "gemini-embedding-2", 3).embed_query("query")
