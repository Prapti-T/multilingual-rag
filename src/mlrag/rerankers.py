"""Optional second-stage reranking of the fused candidate list."""
from __future__ import annotations

import dataclasses
import os
from typing import Protocol

from .embedders import _retry
from .types import Hit


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


def _doc_text(h: Hit) -> str:
    return f"{h.context}\n{h.text}"


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, _doc_text(h)) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda p: float(p[1]), reverse=True)
        return [dataclasses.replace(h, score=float(s)) for h, s in ranked[:top_n]]


class CohereReranker:
    def __init__(self, model: str = "rerank-multilingual-v3.0", api_key: str | None = None):
        import cohere  # optional dependency

        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")
        self.name = model
        self.client = cohere.ClientV2(key)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        resp = _retry(lambda: self.client.rerank(
            model=self.name, query=query, documents=[_doc_text(h) for h in hits], top_n=top_n))
        return [dataclasses.replace(hits[r.index], score=float(r.relevance_score)) for r in resp.results]


def build_reranker(cfg) -> Reranker | None:
    if cfg.provider == "none":
        return None
    if cfg.provider == "cross-encoder":
        return CrossEncoderReranker(cfg.model)
    if cfg.provider == "cohere":
        return CohereReranker(cfg.model if cfg.model.startswith("rerank") else "rerank-multilingual-v3.0")
    raise ValueError(f"Unknown reranker provider: {cfg.provider!r}")
