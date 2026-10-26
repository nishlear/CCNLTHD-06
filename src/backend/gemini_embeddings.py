"""Pinned Gemini embeddings: separate Content objects yield one vector per CV.

Passing a plain list of strings to Embedding 2 aggregates the inputs. Explicit
Content boundaries instead produce one synchronous batchEmbedContents request.
"""

import math


class GeminiEmbeddings:
    def __init__(self, client, model, dimensions):
        self.client, self.model, self.dimensions = client, model.removeprefix("models/"), dimensions
        if self.model != "gemini-embedding-2":
            raise ValueError("Embedding model is pinned to gemini-embedding-2")

    def _normalize(self, values):
        values = [float(value) for value in values]
        if len(values) != self.dimensions or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid Gemini embedding shape or values")
        norm = math.sqrt(sum(value * value for value in values))
        if not norm or not math.isfinite(norm):
            raise ValueError("Gemini returned a zero embedding")
        return [value / norm for value in values]

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model, contents="task: search result | query: " + text,
            config={"output_dimensionality": self.dimensions})
        if not result.embeddings or len(result.embeddings) != 1:
            raise ValueError("Expected one query embedding")
        return self._normalize(result.embeddings[0].values)

    def embed_documents(self, texts):
        if not texts:
            return []
        from google.genai import types
        contents = [types.Content(parts=[types.Part(text="title: none | text: " + text)]) for text in texts]
        result = self.client.models.embed_content(model=self.model, contents=contents,
                                                  config={"output_dimensionality": self.dimensions})
        if not result.embeddings or len(result.embeddings) != len(texts):
            raise ValueError("Expected exactly one embedding per document")
        return [self._normalize(embedding.values) for embedding in result.embeddings]
