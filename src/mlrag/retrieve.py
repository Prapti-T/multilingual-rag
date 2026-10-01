from __future__ import annotations

import time

from .embedders import Embedder
from .rerankers import Reranker
from .sparse import Bm25Encoder
from .store import QdrantStore
from .types import Hit

MODES = ("dense", "sparse", "hybrid")


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, sparse: Bm25Encoder | None,
                 reranker: Reranker | None = None, candidate_k: int = 30, prefetch_k: int = 50):
        self.store, self.embedder, self.sparse = store, embedder, sparse
        self.reranker, self.candidate_k, self.prefetch_k = reranker, candidate_k, prefetch_k

    def retrieve(self, query: str, *, mode: str = "hybrid", top_k: int = 5, lang: str | None = None,
                 rerank: bool = True) -> tuple[list[Hit], dict[str, float]]:
        """Returns (hits, timings). ``rerank=False`` skips the reranker even if one is configured."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode != "dense" and self.sparse is None:
            raise ValueError("This retriever has no sparse encoder; use mode='dense'.")
        use_rerank = rerank and self.reranker is not None
        limit = max(self.candidate_k, top_k) if use_rerank else top_k

        t0 = time.perf_counter()
        dense = self.embedder.embed_queries([query])[0] if mode != "sparse" else None
        t1 = time.perf_counter()
        sparse = self.sparse.encode_query(query) if self.sparse and mode != "dense" else None
        hits = self.store.search(mode=mode, dense=dense, sparse=sparse, limit=limit, lang=lang,
                                 prefetch_k=self.prefetch_k)
        t2 = time.perf_counter()
        timings = {"embed_s": t1 - t0, "search_s": t2 - t1}
        if use_rerank:
            hits = self.reranker.rerank(query, hits, top_k)
            timings["rerank_s"] = time.perf_counter() - t2
        return hits[:top_k], timings
