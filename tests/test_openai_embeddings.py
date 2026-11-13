import json
import math
import unittest
from unittest.mock import Mock, patch

from src.backend.openai_embeddings import OpenAIEmbeddings


def response(vectors):
    body = json.dumps({"data": [{"index": i, "embedding": v} for i, v in reversed(list(enumerate(vectors)))]})
    r = Mock()
    r.read.return_value = body.encode()
    r.__enter__ = Mock(return_value=r)
    r.__exit__ = Mock(return_value=False)
    return r


class OpenAIEmbeddingsTests(unittest.TestCase):
    def model(self):
        return OpenAIEmbeddings("key", "text-embedding-3-small", 1536, 30)

    def test_rejects_models_outside_allowlist(self):
        with self.assertRaises(ValueError):
            OpenAIEmbeddings("key", "text-embedding-ada-002", 1536, 30)
        with self.assertRaises(ValueError):
            OpenAIEmbeddings("key", "text-embedding-3-small", 768, 30)

    def test_documents_keep_order_and_are_normalized(self):
        a, b = [3.0] + [0.0] * 1535, [0.0, 4.0] + [0.0] * 1534
        urlopen = Mock(return_value=response([a, b]))
        with patch("urllib.request.urlopen", urlopen):
            out = self.model().embed_documents(["first", "second"])
        self.assertEqual((out[0][0], out[1][1]), (1.0, 1.0))
        self.assertTrue(all(math.isclose(sum(x * x for x in v), 1.0) for v in out))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(request.get_header("Authorization"), "Bearer key")
        self.assertEqual(json.loads(request.data), {"model": "text-embedding-3-small", "input": ["first", "second"]})

    def test_wrong_shape_is_rejected(self):
        with patch("urllib.request.urlopen", Mock(return_value=response([[1.0, 0.0]]))):
            with self.assertRaises(ValueError):
                self.model().embed_query("q")

    def test_empty_input_makes_no_request(self):
        with patch("urllib.request.urlopen") as urlopen:
            self.assertEqual(self.model().embed_documents([]), [])
        urlopen.assert_not_called()
