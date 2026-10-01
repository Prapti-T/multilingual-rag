"""Documents (JSONL) -> chunks -> dense + sparse vectors -> Qdrant."""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_document
from .config import Config
from .embedders import Embedder
from .sparse import Bm25Encoder
from .store import QdrantStore


def read_docs(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def meta_path(cfg: Config) -> Path:
    return Path(cfg.index.meta_dir) / f"{cfg.qdrant.collection}.json"


def chunk_corpus(docs: list[dict], embedder: Embedder, cfg: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d, embedder.count_tokens, cfg.chunking.max_tokens,
                                     cfg.chunking.overlap_sentences))
    return chunks


def ingest(docs: list[dict], cfg: Config, embedder: Embedder, store: QdrantStore,
           recreate: bool = False, batch_size: int = 64) -> dict:
    t0 = time.time()
    chunks = chunk_corpus(docs, embedder, cfg)
    if not chunks:
        raise ValueError("No chunks produced. Is the documents file empty?")

    encoder = Bm25Encoder(cfg.sparse.k1, cfg.sparse.b, fold_length=cfg.sparse.fold_vowel_length)
    encoder.fit(c.embed_text for c in chunks)
    sparse = [encoder.encode_document(c.embed_text) for c in chunks]

    dense_parts = []
    for i in range(0, len(chunks), batch_size):
        dense_parts.append(embedder.embed_documents([c.embed_text for c in chunks[i : i + batch_size]]))
    dense = np.concatenate(dense_parts)

    store.ensure_collection(dense.shape[1], recreate=recreate)
    store.upsert(chunks, dense, sparse)

    mp = meta_path(cfg)
    mp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"embedder": embedder.name, "dense_dim": int(dense.shape[1]),
            "sparse": encoder.to_dict(), "chunking": vars(cfg.chunking),
            "n_chunks": len(chunks), "n_docs": len(docs)}
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    by_lang: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_lang[c.lang].append(c.n_tokens)
    return {
        "docs": len(docs), "chunks": len(chunks), "seconds": round(time.time() - t0, 1),
        "tokens_per_chunk_by_lang": {
            lang: {"chunks": len(v), "mean": round(statistics.mean(v), 1), "max": max(v)}
            for lang, v in sorted(by_lang.items())
        },
    }


def load_meta(cfg: Config) -> dict:
    mp = meta_path(cfg)
    if not mp.exists():
        raise FileNotFoundError(
            f"No index metadata at {mp}. Run `mlrag ingest` first (it records the sparse-encoder "
            "parameters that queries must reuse).")
    return json.loads(mp.read_text(encoding="utf-8"))
