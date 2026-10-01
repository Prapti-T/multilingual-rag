"""Qdrant store: one collection with a named dense vector and a named sparse vector.

Hybrid search uses Qdrant's Query API: a dense branch and a sparse (BM25) branch are
prefetched and fused server-side with Reciprocal Rank Fusion, so no score calibration
between cosine similarity and BM25 is needed.
"""
from __future__ import annotations

import warnings
from typing import Iterator, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as m

from .chunking import Chunk
from .sparse import SparseVec
from .types import Hit

DENSE, SPARSE = "dense", "sparse"


class QdrantStore:
    def __init__(self, collection: str, url: str | None = None, api_key: str | None = None,
                 path: str | None = ":memory:"):
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        elif path in (None, ":memory:"):
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=path)

    # ---- collection management -------------------------------------------------
    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            params = self.client.get_collection(self.collection).config.params.vectors
            have = params[DENSE].size if isinstance(params, dict) else params.size
            if have != dim:
                raise ValueError(
                    f"Collection '{self.collection}' holds {have}-d vectors but the embedder "
                    f"produces {dim}-d. Use a different collection name or re-ingest with --recreate."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: m.VectorParams(size=dim, distance=m.Distance.COSINE)},
            sparse_vectors_config={SPARSE: m.SparseVectorParams(modifier=m.Modifier.IDF)},
        )
        for field in ("lang", "doc_id"):  # indexes speed up filters on a server; no-ops in embedded mode
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.client.create_payload_index(self.collection, field, m.PayloadSchemaType.KEYWORD)
            except Exception:  # noqa: BLE001
                pass

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ---- writes -------------------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk], dense: np.ndarray, sparse: Sequence[SparseVec],
               batch_size: int = 128) -> None:
        points = []
        for ch, d, (si, sv) in zip(chunks, dense, sparse):
            vector: dict = {DENSE: d.tolist()}
            if si:
                vector[SPARSE] = m.SparseVector(indices=si, values=sv)
            points.append(m.PointStruct(id=ch.chunk_id, vector=vector, payload={
                "doc_id": ch.doc_id, "title": ch.title, "lang": ch.lang, "url": ch.url,
                "heading": ch.heading, "chunk_index": ch.chunk_index, "text": ch.text,
                "context": ch.context, "n_tokens": ch.n_tokens,
            }))
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points[i : i + batch_size])

    # ---- reads --------------------------------------------------------------------
    @staticmethod
    def _hit(p) -> Hit:
        pl = p.payload
        return Hit(id=str(p.id), score=float(p.score), text=pl["text"], context=pl.get("context", ""),
                   doc_id=pl["doc_id"], title=pl["title"], lang=pl["lang"], url=pl.get("url", ""),
                   chunk_index=pl["chunk_index"], heading=pl.get("heading", ""))

    def search(self, *, mode: str, dense: Sequence[float] | None, sparse: SparseVec | None,
               limit: int, lang: str | None = None, prefetch_k: int = 50) -> list[Hit]:
        flt = None
        if lang:
            flt = m.Filter(must=[m.FieldCondition(key="lang", match=m.MatchValue(value=lang))])
        has_sparse = bool(sparse and sparse[0])
        sv = m.SparseVector(indices=sparse[0], values=sparse[1]) if has_sparse else None

        if mode == "hybrid" and not has_sparse:
            mode = "dense"  # query had no lexical terms (e.g. only punctuation)
        if mode == "dense":
            res = self.client.query_points(self.collection, query=list(dense), using=DENSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        elif mode == "sparse":
            if not has_sparse:
                return []
            res = self.client.query_points(self.collection, query=sv, using=SPARSE,
                                           query_filter=flt, limit=limit, with_payload=True)
        elif mode == "hybrid":
            res = self.client.query_points(
                self.collection,
                prefetch=[
                    m.Prefetch(query=list(dense), using=DENSE, limit=prefetch_k, filter=flt),
                    m.Prefetch(query=sv, using=SPARSE, limit=prefetch_k, filter=flt),
                ],
                query=m.FusionQuery(fusion=m.Fusion.RRF),
                limit=limit, with_payload=True,
            )
        else:
            raise ValueError(f"Unknown retrieval mode: {mode!r}")
        return [self._hit(p) for p in res.points]

    def iter_payloads(self, page: int = 256) -> Iterator[dict]:
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, limit=page, offset=offset,
                                                with_payload=True, with_vectors=False)
            for p in points:
                yield p.payload
            if offset is None:
                break
