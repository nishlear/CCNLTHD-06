"""OpenAI embeddings over plain HTTPS (no SDK), one vector per input text.

Vectors are L2-normalized like GeminiEmbeddings so cosine distances in Chroma
are comparable. HTTP errors are raised as urllib.error.HTTPError, so a 429 is
picked up by services._embed_with_backoff through its integer `.code`.
"""

import json
import math
import urllib.request

ALLOWED = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbeddings:
    def __init__(self, api_key, model, dimensions, timeout, base_url="https://api.openai.com/v1"):
        if ALLOWED.get(model) != dimensions:
            raise ValueError(f"Unsupported OpenAI embedding model/dimensions: {model}/{dimensions}")
        self.api_key, self.model, self.dimensions = api_key, model, dimensions
        self.timeout, self.base_url = timeout, base_url.rstrip("/")

    def _normalize(self, values):
        values = [float(value) for value in values]
        if len(values) != self.dimensions or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid OpenAI embedding shape or values")
        norm = math.sqrt(sum(value * value for value in values))
        if not norm or not math.isfinite(norm):
            raise ValueError("OpenAI returned a zero embedding")
        return [value / norm for value in values]

    def _embed(self, texts):
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.last_usage = data.get("usage")  # read by scripts/benchmark.py for cost
        rows = sorted(data.get("data") or [], key=lambda row: row.get("index", 0))
        if len(rows) != len(texts):
            raise ValueError("Expected exactly one embedding per input")
        return [self._normalize(row["embedding"]) for row in rows]

    def embed_query(self, text):
        return self._embed([text])[0]

    def embed_documents(self, texts):
        return self._embed(list(texts)) if texts else []
