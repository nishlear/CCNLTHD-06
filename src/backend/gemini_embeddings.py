"""Text retrieval embeddings using the Google Gen AI SDK.

Embedding 2 requires prompt task prefixes, not the legacy task_type field.
One request per text avoids the SDK's multi-input aggregation semantics.
"""

import math


class GeminiEmbeddings:
    def __init__(self, client, model, dimensions):
        self.client, self.model, self.dimensions = client, model.removeprefix("models/"), dimensions
        if self.model not in {"gemini-embedding-2", "gemini-embedding-001"}:
            raise ValueError("Unsupported Gemini embedding model")

    def _embed(self, text, document):
        options = {"output_dimensionality": self.dimensions}
        if self.model == "gemini-embedding-2":
            text = ("title: none | text: " if document else "task: search result | query: ") + text
        else:
            options["task_type"] = "RETRIEVAL_DOCUMENT" if document else "RETRIEVAL_QUERY"
        result = self.client.models.embed_content(model=self.model, contents=text, config=options)
        if not result.embeddings or len(result.embeddings) != 1:
            raise ValueError("Expected exactly one embedding per text")
        values = [float(value) for value in result.embeddings[0].values]
        if len(values) != self.dimensions or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid Gemini embedding shape or values")
        norm = math.sqrt(sum(value * value for value in values))
        if not norm or not math.isfinite(norm):
            raise ValueError("Gemini returned a zero embedding")
        return [value / norm for value in values]

    def embed_query(self, text):
        return self._embed(text, document=False)

    def embed_documents(self, texts):
        return [self._embed(text, document=True) for text in texts]
