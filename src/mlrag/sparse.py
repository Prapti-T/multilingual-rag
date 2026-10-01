"""BM25-style sparse vectors for Qdrant, with a Unicode-aware analyzer.

Document vectors carry the BM25 term-frequency / length-normalisation weight. Query
vectors are binary, and the collection's ``Modifier.IDF`` multiplies in the IDF
server-side, so dot product == BM25 and IDF stays correct as the corpus grows.

Terms are hashed to u32 indices (crc32): no vocabulary file to ship, and the mapping is
stable across runs and machines. Collisions are negligible at this corpus size.
"""
from __future__ import annotations

import zlib
from collections import Counter
from typing import Iterable

from .text import tokenize_words

SparseVec = tuple[list[int], list[float]]


class Bm25Encoder:
    def __init__(
        self, k1: float = 1.2, b: float = 0.75, avg_len: float = 64.0, fold_length: bool = True
    ):
        self.k1, self.b, self.avg_len, self.fold_length = k1, b, avg_len, fold_length

    def tokens(self, text: str) -> list[str]:
        return tokenize_words(text, fold_length=self.fold_length)

    @staticmethod
    def _index(token: str) -> int:
        return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF

    def fit(self, texts: Iterable[str]) -> "Bm25Encoder":
        lengths = [len(self.tokens(t)) for t in texts]
        lengths = [n for n in lengths if n > 0]
        self.avg_len = sum(lengths) / len(lengths) if lengths else 1.0
        return self

    def encode_document(self, text: str) -> SparseVec:
        toks = self.tokens(text)
        if not toks:
            return [], []
        dl = len(toks)
        tf = Counter(self._index(t) for t in toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avg_len)
        items = sorted(tf.items())
        return [i for i, _ in items], [c * (self.k1 + 1) / (c + norm) for _, c in items]

    def encode_query(self, text: str) -> SparseVec:
        idx = sorted({self._index(t) for t in self.tokens(text)})
        return idx, [1.0] * len(idx)

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "avg_len": self.avg_len, "fold_length": self.fold_length}

    @classmethod
    def from_dict(cls, d: dict) -> "Bm25Encoder":
        return cls(d["k1"], d["b"], d["avg_len"], d["fold_length"])
