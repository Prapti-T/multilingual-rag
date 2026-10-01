import pytest

from mlrag.ingest import ingest
from mlrag.retrieve import Retriever


def top_doc(hits):
    return hits[0].doc_id if hits else None


def test_ingest_reports_stats_per_language(index):
    assert index.stats["docs"] == 5 and index.stats["chunks"] == 5
    assert set(index.stats["tokens_per_chunk_by_lang"]) == {"ne", "en"}
    assert index.store.count() == 5


def test_reingest_is_idempotent(index):
    from tests.conftest import DOCS  # noqa: PLC0415

    ingest(DOCS, index.cfg, index.embedder, index.store)
    assert index.store.count() == 5


def test_sparse_retrieval_finds_exact_devanagari_terms(index):
    hits, _ = index.retriever.retrieve("नेपालको राजधानी", mode="sparse", top_k=3)
    assert top_doc(hits) == "ne:काठमाडौं"


def test_vowel_length_folding_rescues_misspelled_query(index, index_no_fold):
    # "राजधानि" (short i) never appears in the corpus, which spells it "राजधानी".
    hits, _ = index.retriever.retrieve("राजधानि", mode="sparse", top_k=3)
    assert top_doc(hits) == "ne:काठमाडौं"
    hits, _ = index_no_fold.retriever.retrieve("राजधानि", mode="sparse", top_k=3)
    assert hits == []  # without folding the lexical branch has nothing to match


def test_hybrid_and_dense_find_english_passage(index):
    for mode in ("dense", "hybrid"):
        hits, timings = index.retriever.retrieve("photosynthesis plants sunlight", mode=mode, top_k=3)
        assert top_doc(hits) == "en:Photosynthesis"
        assert {"embed_s", "search_s"} <= set(timings)


def test_language_filter_restricts_results(index):
    hits, _ = index.retriever.retrieve("highest mountain Nepal", mode="hybrid", top_k=5, lang="ne")
    assert hits and all(h.lang == "ne" for h in hits)
    hits, _ = index.retriever.retrieve("highest mountain Nepal", mode="hybrid", top_k=5, lang="en")
    assert hits and all(h.lang == "en" for h in hits)


def test_hybrid_falls_back_to_dense_when_query_has_no_terms(index):
    hits, _ = index.retriever.retrieve("???", mode="hybrid", top_k=3)
    assert len(hits) == 3
    hits, _ = index.retriever.retrieve("???", mode="sparse", top_k=3)
    assert hits == []


def test_embedding_dimension_mismatch_is_caught(index):
    with pytest.raises(ValueError, match="re-ingest with --recreate"):
        index.store.ensure_collection(index.embedder.dim + 1)


def test_recreate_rebuilds_collection(index):
    from tests.conftest import DOCS  # noqa: PLC0415

    ingest(DOCS[:2], index.cfg, index.embedder, index.store, recreate=True)
    assert index.store.count() == 2


def test_invalid_mode_and_missing_sparse_encoder(index):
    with pytest.raises(ValueError):
        index.retriever.retrieve("x", mode="bm25")
    dense_only = Retriever(index.store, index.embedder, sparse=None)
    with pytest.raises(ValueError, match="no sparse encoder"):
        dense_only.retrieve("x", mode="hybrid")
    assert dense_only.retrieve("नेपाल", mode="dense", top_k=2)[0]


class ReverseReranker:
    name = "reverse"

    def rerank(self, query, hits, top_n):
        return list(reversed(hits))[:top_n]


def test_reranker_runs_on_candidates_and_can_be_bypassed(index):
    r = Retriever(index.store, index.embedder, index.sparse, ReverseReranker(), candidate_k=5)
    plain, _ = r.retrieve("Mount Everest highest", mode="hybrid", top_k=2, rerank=False)
    reranked, timings = r.retrieve("Mount Everest highest", mode="hybrid", top_k=2, rerank=True)
    assert "rerank_s" in timings
    assert [h.id for h in reranked] != [h.id for h in plain]
    assert top_doc(plain) == "en:Mount Everest"
