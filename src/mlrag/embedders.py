"""Embedding backends behind one small interface.

* ``sentence-transformers``: any local open-weight model (BGE-M3, multilingual-e5, ...).
* ``cohere``: hosted; a free trial key is enough for a corpus this size.
* ``hash``: deterministic, offline, NOT semantic. Exists so the plumbing can be tested
  without downloading a model. Never use it for results.
"""
from __future__ import annotations

import os
import time
import zlib
from typing import Callable, Protocol, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def count_tokens(self, text: str) -> int: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def _retry(fn: Callable[[], T], attempts: int = 6) -> T:
    delay = 2.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - SDK-specific exception types vary
            status = getattr(e, "status_code", None)
            if i == attempts - 1 or status not in (429, 500, 502, 503, 504):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        max_seq_length: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size
        self._tok = self.model.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def _encode(self, texts: Sequence[str], prefix: str, method: str) -> np.ndarray:
        # Prefer the model's own query/document prompts (sentence-transformers >= 5)
        # unless the config sets explicit prefixes.
        fn = getattr(self.model, method, None)
        if fn is not None and not prefix:
            vecs = fn(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                      show_progress_bar=len(texts) > 256)
        else:
            vecs = self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=len(texts) > 256)
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.doc_prefix, "encode_document")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix, "encode_query")


class CohereEmbedder:
    def __init__(self, model: str = "embed-multilingual-v3.0", api_key: str | None = None,
                 batch_size: int = 96):
        import cohere  # optional dependency

        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self.name = model
        self.client = cohere.ClientV2(key)
        self.batch_size = batch_size

    def count_tokens(self, text: str) -> int:
        # Approximation: the hosted tokenizer is not available locally. Deliberately
        # conservative for Devanagari, which tokenizes far less efficiently than English.
        return max(len(text.split()) * 2, len(text) // 3, 1)

    def _embed(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = _retry(lambda b=batch: self.client.embed(
                model=self.name, input_type=input_type, texts=b, embedding_types=["float"]))
            out.extend(resp.embeddings.float_)
        return _l2(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "search_document")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "search_query")


class HashEmbedder:
    """Hashed character 3-grams. Lexical, deterministic, offline. For tests only."""

    def __init__(self, dim: int = 256):
        self.name = f"hash-{dim}"
        self.dim = dim

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {text.casefold()} "
        for i in range(len(t) - 2):
            v[zlib.crc32(t[i : i + 3].encode("utf-8")) % self.dim] += 1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts]))

    embed_queries = embed_documents


def build_embedder(cfg) -> Embedder:
    p = cfg.provider
    if p == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg.model, cfg.query_prefix, cfg.doc_prefix,
                                           cfg.max_seq_length, cfg.batch_size)
    if p == "cohere":
        return CohereEmbedder(cfg.model)
    if p == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding provider: {p!r}")
