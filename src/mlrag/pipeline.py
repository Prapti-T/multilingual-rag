from __future__ import annotations

import time
from typing import Sequence

from .config import Config
from .embedders import Embedder, build_embedder
from .generate import Generator, build_generator
from .ingest import load_meta
from .rerankers import Reranker, build_reranker
from .retrieve import Retriever
from .sparse import Bm25Encoder
from .store import QdrantStore
from .types import Answer


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, cfg: Config):
        self.retriever, self.generator, self.cfg = retriever, generator, cfg

    @classmethod
    def from_config(cls, cfg: Config) -> "RAGPipeline":
        meta = load_meta(cfg)
        embedder: Embedder = build_embedder(cfg.embedding)
        if meta["embedder"] != embedder.name:
            raise ValueError(f"Index was built with '{meta['embedder']}' but config selects "
                             f"'{embedder.name}'. Query and document embedders must match.")
        sparse = Bm25Encoder.from_dict(meta["sparse"]) if cfg.sparse.enabled else None
        store = QdrantStore(cfg.qdrant.collection, cfg.qdrant.url, cfg.qdrant.api_key, cfg.qdrant.path)
        reranker: Reranker | None = build_reranker(cfg.reranker)
        retriever = Retriever(store, embedder, sparse, reranker, cfg.retrieval.candidate_k,
                              cfg.retrieval.prefetch_k)
        return cls(retriever, build_generator(cfg.generation), cfg)

    def ask(self, question: str, *, history: Sequence[dict] | None = None, mode: str | None = None,
            top_k: int | None = None, lang: str | None = None) -> Answer:
        mode = mode or self.cfg.retrieval.mode
        top_k = top_k or self.cfg.retrieval.top_k
        hits, timings = self.retriever.retrieve(question, mode=mode, top_k=top_k, lang=lang)
        t0 = time.perf_counter()
        gen = self.generator.generate(question, hits, history)
        timings["generate_s"] = time.perf_counter() - t0
        timings["total_s"] = sum(timings.values())
        return Answer(
            question=question, answer=gen.text, abstained=gen.abstained, mode=mode,
            citations=[hits[n - 1] for n in gen.cited], retrieved=hits,
            invalid_citations=gen.invalid,
            query_tokens=self.retriever.embedder.count_tokens(question),
            timings={k: round(v, 4) for k, v in timings.items()},
        )
