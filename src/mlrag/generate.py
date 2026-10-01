"""Grounded answer generation with numbered citations and explicit abstention.

The model is asked to emit the sentinel ``[NO_ANSWER]`` when the passages do not contain
the answer. That gives a machine-readable abstention signal for evaluation. A small model
may ignore the instruction, so label a sample by hand before trusting the rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx

from .types import Hit

NO_ANSWER = "[NO_ANSWER]"
ABSTAIN_TEXT = "I could not find this in the provided sources. / स्रोतहरूमा यसको उत्तर भेटिएन।"

SYSTEM_PROMPT = (
    "You are a study assistant. Answer from the numbered source passages only.\n"
    "1. Use ONLY the passages. Do not use outside knowledge.\n"
    "2. After each claim, cite the passage it came from as [1], [2], ...\n"
    "3. Reply in the same language and script as the user's question.\n"
    f"4. If the passages do not contain the answer, reply with exactly {NO_ANSWER} and nothing else.\n"
    "5. Keep the answer short."
)

_CITE = re.compile(r"\[(\d+)\]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Generation:
    text: str
    abstained: bool
    cited: list[int]  # valid 1-based passage numbers, in order of first citation
    invalid: list[int]  # cited numbers with no matching passage


def build_context(hits: Sequence[Hit], max_chars: int) -> str:
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[{i}] ({h.context}; {h.lang})\n{h.text}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_messages(question: str, hits: Sequence[Hit], history: Sequence[dict] | None,
                   max_context_chars: int, history_turns: int) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        msgs.extend({"role": t["role"], "content": t["content"]}
                    for t in list(history)[-2 * history_turns:] if t.get("role") in ("user", "assistant"))
    ctx = build_context(hits, max_context_chars)
    msgs.append({"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"})
    return msgs


def parse_generation(raw: str, n_passages: int) -> Generation:
    text = _THINK.sub("", raw).strip()  # some local models emit <think> blocks
    if NO_ANSWER in text:
        return Generation(ABSTAIN_TEXT, True, [], [])
    cited, invalid = [], []
    for n in (int(x) for x in _CITE.findall(text)):
        bucket = cited if 1 <= n <= n_passages else invalid
        if n not in bucket:
            bucket.append(n)
    return Generation(text, False, cited, invalid)


class Generator(Protocol):
    def generate(self, question: str, hits: Sequence[Hit], history: Sequence[dict] | None) -> Generation: ...


class ExtractiveGenerator:
    """No LLM: returns the best passage with a citation. Free, deterministic, retrieval-only."""

    def __init__(self, max_chars: int = 600):
        self.max_chars = max_chars

    def generate(self, question, hits, history=None) -> Generation:
        if not hits:
            return Generation(ABSTAIN_TEXT, True, [], [])
        body = hits[0].text
        if len(body) > self.max_chars:
            body = body[: self.max_chars].rstrip() + "…"
        return Generation(f"{body} [1]", False, [1], [])


class _ChatGenerator:
    def __init__(self, model: str, base_url: str, temperature: float, max_context_chars: int,
                 history_turns: int, timeout_s: float, client: httpx.Client | None = None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.max_context_chars = temperature, max_context_chars
        self.history_turns = history_turns
        self.client = client or httpx.Client(timeout=timeout_s)

    def _chat(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def generate(self, question, hits, history=None) -> Generation:
        msgs = build_messages(question, hits, history, self.max_context_chars, self.history_turns)
        return parse_generation(self._chat(msgs), len(hits))


class OllamaGenerator(_ChatGenerator):
    def _chat(self, messages: list[dict]) -> str:
        r = self.client.post(f"{self.base_url}/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": self.temperature}})
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatGenerator(_ChatGenerator):
    """Any OpenAI-compatible /chat/completions endpoint (hosted free tiers, vLLM, LM Studio, ...)."""

    def __init__(self, *args, api_key: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key

    def _chat(self, messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self.client.post(f"{self.base_url}/chat/completions", headers=headers, json={
            "model": self.model, "messages": messages, "temperature": self.temperature})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def build_generator(cfg) -> Generator:
    if cfg.provider == "extractive":
        return ExtractiveGenerator()
    common = dict(model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature,
                  max_context_chars=cfg.max_context_chars, history_turns=cfg.history_turns,
                  timeout_s=cfg.timeout_s)
    if cfg.provider == "ollama":
        return OllamaGenerator(**common)
    if cfg.provider == "openai-compat":
        return OpenAICompatGenerator(**common, api_key=cfg.api_key)
    raise ValueError(f"Unknown generation provider: {cfg.provider!r}")
