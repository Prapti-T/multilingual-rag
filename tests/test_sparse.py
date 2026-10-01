from mlrag.sparse import Bm25Encoder


def weight_of(enc, text, term):
    idx, val = enc.encode_document(text)
    return dict(zip(idx, val))[Bm25Encoder._index(term)]


def test_term_frequency_saturates():
    enc = Bm25Encoder(avg_len=3.0)
    w1 = weight_of(enc, "cat dog bird", "cat")
    w2 = weight_of(enc, "cat cat dog", "cat")
    w10 = weight_of(enc, "cat " * 10 + "x", "cat")
    assert w1 < w2 < w10 < enc.k1 + 1.0  # grows with tf but is bounded by k1 + 1


def test_longer_documents_are_penalised_for_the_same_tf():
    enc = Bm25Encoder(avg_len=5.0)
    short = weight_of(enc, "cat a b c d", "cat")
    long = weight_of(enc, "cat " + "filler " * 40, "cat")
    assert long < short


def test_indices_sorted_deterministic_and_unique():
    enc = Bm25Encoder()
    a = enc.encode_document("beta alpha beta gamma")
    b = enc.encode_document("beta alpha beta gamma")
    assert a == b
    assert a[0] == sorted(set(a[0]))


def test_query_vector_is_binary_and_deduplicated():
    idx, val = Bm25Encoder().encode_query("cat cat dog")
    assert len(idx) == 2 and val == [1.0, 1.0]


def test_empty_text_gives_empty_vectors():
    enc = Bm25Encoder()
    assert enc.encode_document("... !!!") == ([], [])
    assert enc.encode_query("") == ([], [])


def test_vowel_length_folding_makes_variants_match():
    folded = Bm25Encoder(fold_length=True)
    plain = Bm25Encoder(fold_length=False)
    assert folded.encode_query("राजधानि")[0] == folded.encode_document("राजधानी")[0]
    assert plain.encode_query("राजधानि")[0] != plain.encode_document("राजधानी")[0]


def test_fit_sets_average_length_and_roundtrips():
    enc = Bm25Encoder().fit(["a b c d", "a b"])
    assert enc.avg_len == 3.0
    assert Bm25Encoder.from_dict(enc.to_dict()).to_dict() == enc.to_dict()
