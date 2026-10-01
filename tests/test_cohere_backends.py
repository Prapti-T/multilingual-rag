"""Offline checks of the Cohere adapters against the SDK's *real* response types (no network, no key)."""
import numpy as np
import pytest

cohere = pytest.importorskip("cohere")

from cohere.types import EmbedByTypeResponse, EmbedByTypeResponseEmbeddings  # noqa: E402
from cohere.v2.types import V2RerankResponse, V2RerankResponseResultsItem  # noqa: E402

from mlrag.embedders import CohereEmbedder  # noqa: E402
from mlrag.rerankers import CohereReranker  # noqa: E402
from mlrag.types import Hit  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def embed(self, **kw):
        self.calls.append(kw)
        n = len(kw["texts"])
        vecs = [[3.0, 4.0]] * n  # not unit length: the adapter must normalise
        return EmbedByTypeResponse(id="x", embeddings=EmbedByTypeResponseEmbeddings(float_=vecs),
                                   response_type="embeddings_by_type")

    def rerank(self, **kw):
        self.calls.append(kw)
        return V2RerankResponse(results=[V2RerankResponseResultsItem(index=1, relevance_score=0.9),
                                         V2RerankResponseResultsItem(index=0, relevance_score=0.2)])


def hit(i):
    return Hit(id=str(i), score=0.0, text=f"t{i}", context=f"c{i}", doc_id=f"d{i}", title="", lang="ne",
               url="", chunk_index=i)


def test_embedder_batches_normalises_and_sets_input_types(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key-not-real")
    emb = CohereEmbedder(batch_size=2)
    emb.client = FakeClient()
    out = emb.embed_documents(["a", "b", "c"])
    assert out.shape == (3, 2) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert [len(c["texts"]) for c in emb.client.calls] == [2, 1]
    assert {c["input_type"] for c in emb.client.calls} == {"search_document"}
    emb.embed_queries(["q"])
    assert emb.client.calls[-1]["input_type"] == "search_query"
    assert emb.client.calls[-1]["embedding_types"] == ["float"]


def test_reranker_maps_indices_back_to_hits(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key-not-real")
    rr = CohereReranker()
    rr.client = FakeClient()
    out = rr.rerank("query", [hit(0), hit(1)], top_n=2)
    assert [h.id for h in out] == ["1", "0"] and out[0].score == 0.9
    assert rr.client.calls[0]["documents"] == ["c0\nt0", "c1\nt1"]


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        CohereEmbedder()
