"""Sentence-aware chunking with a token budget.

The budget is measured with the *embedding model's own tokenizer* (pass
``embedder.count_tokens``). Chunking with one tokenizer and embedding with another is
a silent way to overflow the embedder's context or to waste it. Devanagari fragments
into many more tokens per word than English, so word- or character-based budgets
are not interchangeable across languages either.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from .text import normalize, split_sentences

CountTokens = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    lang: str
    url: str
    heading: str
    chunk_index: int
    text: str
    n_tokens: int

    @property
    def context(self) -> str:
        return f"{self.title} > {self.heading}" if self.heading else self.title

    @property
    def embed_text(self) -> str:
        """What gets embedded and BM25-indexed: the text with its title/heading prepended."""
        return f"{self.context}\n{self.text}"


def _split_long(sentence: str, count: CountTokens, max_tokens: int) -> list[str]:
    """Hard-split a sentence that alone exceeds the budget, on word boundaries."""
    if count(sentence) <= max_tokens:
        return [sentence]
    pieces: list[str] = []
    words = sentence.split()
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and count(trial) > max_tokens:
            pieces.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def pack_sentences(
    sentences: Iterable[str], count: CountTokens, max_tokens: int, overlap_sentences: int
) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        for piece in _split_long(sent, count, max_tokens):
            t = count(piece)
            if cur and cur_tokens + t > max_tokens:
                chunks.append(" ".join(cur))
                cur = cur[-overlap_sentences:] if overlap_sentences > 0 else []
                cur_tokens = sum(count(x) for x in cur)
                if cur_tokens + t > max_tokens:  # overlap would overflow: drop it
                    cur, cur_tokens = [], 0
            cur.append(piece)
            cur_tokens += t
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(
    doc: dict, count_tokens: CountTokens, max_tokens: int = 256, overlap_sentences: int = 1
) -> list[Chunk]:
    """Chunk one document dict (see fetch.py for the schema) section by section.

    Chunks never cross a section boundary, and ids are deterministic
    (uuid5 of doc_id + position), so re-ingesting the same corpus is idempotent.
    """
    count = lru_cache(maxsize=None)(count_tokens)
    chunks: list[Chunk] = []
    idx = 0
    for section in doc["sections"]:
        text = normalize(section["text"])
        sentences = [s for para in text.split("\n") for s in split_sentences(para)]
        for body in pack_sentences(sentences, count, max_tokens, overlap_sentences):
            if not body.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['doc_id']}#{idx}")),
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    lang=doc["lang"],
                    url=doc.get("url", ""),
                    heading=section.get("heading", ""),
                    chunk_index=idx,
                    text=body,
                    n_tokens=count(body),
                )
            )
            idx += 1
    return chunks
