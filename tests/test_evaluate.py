import csv
import json

import pytest

from mlrag.evaluate import (Question, bootstrap_ci, build_generation_sheet, evaluate_retrieval,
                            first_relevant_rank, format_markdown, is_relevant, load_questions,
                            make_template, report_labels, summarize_ranks, validate_questions)
from mlrag.types import Hit

KATHMANDU = {"doc_id": "ne:काठमाडौं", "span": "नेपालको राजधानी हो"}


def hit(doc_id, text):
    return Hit(id=doc_id + text[:3], score=1.0, text=text, context="", doc_id=doc_id, title="",
               lang="ne", url="", chunk_index=0)


def test_relevance_requires_doc_and_span_with_normalization():
    gold = [{"doc_id": "en:A", "span": "Highest  Mountain"}]
    assert is_relevant("en:A", "It is the highest mountain in the world.", gold)
    assert not is_relevant("en:B", "It is the highest mountain in the world.", gold)
    assert not is_relevant("en:A", "Something unrelated.", gold)


def test_first_relevant_rank():
    gold = [{"doc_id": "d", "span": "needle"}]
    hits = [hit("d", "hay"), hit("x", "needle"), hit("d", "a needle here")]
    assert first_relevant_rank(hits, gold) == 3
    assert first_relevant_rank(hits[:2], gold) is None


def test_summarize_ranks():
    s = summarize_ranks([1, 3, None, 6], (1, 3, 5))
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 1 / 3) / 4)  # rank 6 is beyond max k and not credited


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    vals = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    lo, hi = bootstrap_ci(vals)
    assert (lo, hi) == bootstrap_ci(vals)
    assert lo <= sum(vals) / len(vals) <= hi and 0 <= lo and hi <= 1
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)


def questions():
    return [
        Question("q1", {"ne": "नेपालको राजधानी के हो?", "ne_typo": "नेपालको राजधानि के हो?"}, [KATHMANDU], group="geo"),
        Question("q2", {"en": "What is the highest mountain?"},
                 [{"doc_id": "en:Mount Everest", "span": "highest mountain in the world"}], group="geo"),
        Question("q3", {"en": "Who won the 1990 cricket final?"}, [], answerable=False, group="unanswerable"),
    ]


def test_validate_questions_flags_each_problem_type():
    chunks = {"ne:काठमाडौं": ["काठमाडौं नेपालको राजधानी हो।"]}
    qs = [Question("a", {"en": "x"}, [{"doc_id": "ne:काठमाडौं", "span": "not in the text"}]),
          Question("a", {"en": "x"}, [KATHMANDU]),
          Question("b", {"en": "x"}, [{"doc_id": "missing:doc", "span": "x"}]),
          Question("c", {}, [KATHMANDU]),
          Question("d", {"en": "x"}, [], answerable=True),
          Question("e", {"en": "x"}, [KATHMANDU], answerable=False)]
    text = "\n".join(validate_questions(qs, chunks))
    for needle in ("duplicate id", "span not found", "not in the index", "no non-empty variants",
                   "answerable but has no gold", "marked unanswerable but has gold"):
        assert needle in text
    assert validate_questions([qs[1]], chunks) == []


def test_make_template_balances_languages_and_prefills_gold(index):
    rows = make_template(index.store.iter_payloads(), n=4, n_unanswerable=2, seed=1, min_tokens=5)
    assert len(rows) == 4 + 2
    answerable = [r for r in rows if r["answerable"]]
    assert {r["_passage_lang"] for r in answerable} == {"ne", "en"}
    assert all(r["gold"][0]["span"] in r["_passage"] for r in answerable)
    assert all(not r["gold"] for r in rows if not r["answerable"])
    assert make_template(index.store.iter_payloads(), 4, 2, seed=1, min_tokens=5) == rows


def test_load_questions_ignores_underscore_keys_and_empty_variants(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": "q", "variants": {"en": "hi", "ne": "  "}, "gold": [KATHMANDU],
                             "_passage": "hint"}, ensure_ascii=False) + "\n", encoding="utf-8")
    (q,) = load_questions(p)
    assert q.variants == {"en": "hi"} and q.answerable and "_passage" not in q.extra


def test_evaluate_retrieval_end_to_end_with_paired_delta(index):
    rows = evaluate_retrieval(index.retriever, questions(), ["sparse", "hybrid"], ks=(1, 3, 5),
                              reference_variant="ne")
    by = {(r["mode"], r["variant"]): r for r in rows}
    assert set(by) == {("sparse", "ne"), ("sparse", "ne_typo"), ("sparse", "en"),
                       ("hybrid", "ne"), ("hybrid", "ne_typo"), ("hybrid", "en")}
    assert by[("sparse", "ne_typo")]["hit@1"] == 1.0  # vowel-length folding handles the typo
    assert by[("sparse", "ne_typo")]["delta_hit@5_vs_ne"] == 0.0
    assert by[("hybrid", "en")]["n"] == 1  # unanswerable q3 is excluded
    lo, hi = by[("hybrid", "ne")]["hit@5_ci95"]
    assert lo <= by[("hybrid", "ne")]["hit@5"] <= hi
    md = format_markdown(rows, (1, 3, 5))
    assert "hit@5 95% CI" in md and "delta_hit@5_vs_ne" in md


def test_modes_with_rerank_need_a_reranker(index):
    with pytest.raises(ValueError, match="needs a reranker"):
        evaluate_retrieval(index.retriever, questions(), ["hybrid+rerank"])
    with pytest.raises(ValueError, match="Bad mode spec"):
        evaluate_retrieval(index.retriever, questions(), ["hybrid+magic"])


def test_generation_sheet_hides_variant_and_report_joins_labels_back(index, tmp_path):
    full, sheet = build_generation_sheet(index.pipeline, questions(), ["ne", "en"], tmp_path, mode="hybrid")
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # q1:ne, q2:en, q3:en (q1 has no 'en' variant)
    assert "variant" not in rows[0] and "label" in rows[0]
    for r in rows:
        r["label"] = "correct"
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    report = report_labels(full, sheet)
    assert "| en | 2 |" in report and "| ne | 1 |" in report
    assert "non-existent passage: 0/3" in report
