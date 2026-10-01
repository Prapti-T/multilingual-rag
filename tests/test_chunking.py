import re

from mlrag.chunking import chunk_document


def count(s: str) -> int:
    return len(s.split())


def make_doc(sections):
    return {"doc_id": "en:T", "title": "T", "lang": "en", "url": "u", "sections": sections}


SENTS = [f"Sentence number {i} is here." for i in range(30)]  # 5 "tokens" each


def test_chunks_respect_budget_overlap_and_cover_everything():
    doc = make_doc([{"heading": "H", "text": " ".join(SENTS)}])
    chunks = chunk_document(doc, count, max_tokens=20, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 20 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):  # last sentence of a chunk opens the next one
        last = re.findall(r"Sentence number \d+ is here\.", a.text)[-1]
        assert b.text.startswith(last)
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in SENTS)


def test_overlap_zero_means_disjoint_chunks():
    doc = make_doc([{"heading": "H", "text": " ".join(SENTS)}])
    chunks = chunk_document(doc, count, max_tokens=20, overlap_sentences=0)
    total = sum(c.text.count("Sentence number") for c in chunks)
    assert total == len(SENTS)


def test_chunks_do_not_cross_sections_and_carry_heading_context():
    doc = make_doc([{"heading": "A", "text": "Alpha one is here. Alpha two is here."},
                    {"heading": "B", "text": "Beta one is here. Beta two is here."}])
    chunks = chunk_document(doc, count, max_tokens=100)
    assert [c.heading for c in chunks] == ["A", "B"]
    assert "Beta" not in chunks[0].text
    assert chunks[1].context == "T > B"
    assert chunks[1].embed_text.startswith("T > B\n")


def test_oversized_sentence_is_hard_split_within_budget():
    long_sentence = " ".join(f"w{i}" for i in range(100)) + "."
    chunks = chunk_document(make_doc([{"heading": "", "text": long_sentence}]), count, max_tokens=20)
    assert len(chunks) >= 5
    assert all(c.n_tokens <= 20 for c in chunks)


def test_ids_are_deterministic_and_unique():
    doc = make_doc([{"heading": "H", "text": " ".join(SENTS)}])
    a = chunk_document(doc, count, max_tokens=20)
    b = chunk_document(doc, count, max_tokens=20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_devanagari_danda_sentences_pack_correctly():
    text = " ".join(f"यो वाक्य नम्बर {i} हो।" for i in range(12))
    chunks = chunk_document(make_doc([{"heading": "", "text": text}]), count, max_tokens=10,
                            overlap_sentences=0)
    assert all(c.n_tokens <= 10 for c in chunks)
    assert sum(c.text.count("हो।") for c in chunks) == 12


def test_empty_sections_produce_no_chunks():
    assert chunk_document(make_doc([{"heading": "H", "text": "   \n  "}]), count) == []
