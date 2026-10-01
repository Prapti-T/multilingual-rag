from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Hit:
    """One retrieved chunk."""

    id: str
    score: float
    text: str
    context: str  # "Title > Heading", used for display and embedded together with the text
    doc_id: str
    title: str
    lang: str
    url: str
    chunk_index: int
    heading: str = ""

    def to_dict(self, max_chars: int | None = None) -> dict[str, Any]:
        d = asdict(self)
        if max_chars is not None and len(d["text"]) > max_chars:
            d["text"] = d["text"][:max_chars].rstrip() + "…"
        return d


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    mode: str
    citations: list[Hit] = field(default_factory=list)  # passages the answer actually cites
    retrieved: list[Hit] = field(default_factory=list)  # everything handed to the generator
    invalid_citations: list[int] = field(default_factory=list)  # [n] markers with no such passage
    query_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self, max_chars: int | None = 400) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "mode": self.mode,
            "citations": [h.to_dict(max_chars) for h in self.citations],
            "retrieved": [h.to_dict(max_chars) for h in self.retrieved],
            "invalid_citations": self.invalid_citations,
            "query_tokens": self.query_tokens,
            "timings": self.timings,
        }
