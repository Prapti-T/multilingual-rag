"""Configuration: YAML file -> typed dataclasses, with env-var overrides for secrets/endpoints."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QdrantCfg:
    url: str | None = None  # e.g. http://localhost:6333 (docker compose). None -> embedded local mode
    api_key: str | None = None
    path: str = "./data/qdrant_local"  # embedded mode only; ":memory:" for tests
    collection: str = "mlrag_wiki"


@dataclass
class EmbeddingCfg:
    provider: str = "sentence-transformers"  # sentence-transformers | cohere | hash (offline tests only)
    model: str = "BAAI/bge-m3"
    query_prefix: str = ""  # multilingual-e5 wants "query: " / "passage: "
    doc_prefix: str = ""
    max_seq_length: int | None = 512
    batch_size: int = 32


@dataclass
class ChunkingCfg:
    max_tokens: int = 256
    overlap_sentences: int = 1


@dataclass
class SparseCfg:
    enabled: bool = True
    k1: float = 1.2
    b: float = 0.75
    fold_vowel_length: bool = True  # collapse hrasva/dirgha spelling variation in the lexical index


@dataclass
class RetrievalCfg:
    mode: str = "hybrid"  # dense | sparse | hybrid
    top_k: int = 5
    candidate_k: int = 30  # how many go to the reranker (when one is configured)
    prefetch_k: int = 50  # per-branch depth before RRF fusion


@dataclass
class RerankerCfg:
    provider: str = "none"  # none | cross-encoder | cohere
    model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class GenerationCfg:
    provider: str = "extractive"  # extractive (no LLM) | ollama | openai-compat
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = 0.0
    max_context_chars: int = 6000
    history_turns: int = 3
    timeout_s: float = 120.0


@dataclass
class IndexCfg:
    meta_dir: str = "./data/index"  # sparse-encoder parameters and the embedder used at ingest time


@dataclass
class Config:
    qdrant: QdrantCfg = field(default_factory=QdrantCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    sparse: SparseCfg = field(default_factory=SparseCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    reranker: RerankerCfg = field(default_factory=RerankerCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    index: IndexCfg = field(default_factory=IndexCfg)


def _build(cls: type, data: dict[str, Any], path: str = "") -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{path or 'root'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        ftype = known[name].default_factory  # type: ignore[attr-defined]
        if value is not None and callable(ftype) and is_dataclass(ftype):
            kwargs[name] = _build(ftype, value or {}, f"{path}.{name}".lstrip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_env_file(path: str = ".env") -> None:
    """Minimal .env reader. Existing environment variables win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_ENV_OVERRIDES = {
    "QDRANT_URL": ("qdrant", "url"),
    "QDRANT_API_KEY": ("qdrant", "api_key"),
    "LLM_BASE_URL": ("generation", "base_url"),
    "LLM_API_KEY": ("generation", "api_key"),
    "LLM_MODEL": ("generation", "model"),
}


def load_config(path: str | None = None) -> Config:
    load_env_file()
    path = path or os.environ.get("MLRAG_CONFIG") or "configs/default.yaml"
    data: dict[str, Any] = {}
    if Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg: Config = _build(Config, data)
    for env, (section, key) in _ENV_OVERRIDES.items():
        if os.environ.get(env):
            setattr(getattr(cfg, section), key, os.environ[env])
    return cfg
