"""Retrieval evaluation across query variants (script / spelling / language) and retrieval modes.

Gold is span-based, not id-based. A question lists acceptable ``(doc_id, span)`` pairs and a
retrieved chunk counts as relevant if it comes from that document and contains the span. Chunk
ids change whenever chunking changes; spans do not, so the same question set can be reused while
you vary chunk size, embedder or analyzer.

Question file (JSONL, one object per line):

    {"id": "q001", "group": "science",
     "variants": {"en": "...", "ne": "...", "ne_rom": "...", "ne_typo": "..."},
     "gold": [{"doc_id": "ne:प्रकाश संश्लेषण", "span": "a short verbatim fragment"}],
     "answerable": true}

Unanswerable questions (``"answerable": false``, empty ``gold``) are skipped by the retrieval
metrics and used to measure abstention in the generation sheet. Keys starting with "_" are ignored.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .retrieve import Retriever
from .text import canonical_for_match, split_sentences
from .types import Hit


@dataclass
class Question:
    id: str
    variants: dict[str, str]
    gold: list[dict]
    answerable: bool = True
    group: str = ""
    extra: dict = field(default_factory=dict)


def load_questions(path: str | Path) -> list[Question]:
    out: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            gold = d.get("gold", [])
            out.append(Question(
                id=d["id"], variants={k: v for k, v in d.get("variants", {}).items() if v.strip()},
                gold=gold, answerable=d.get("answerable", bool(gold)), group=d.get("group", ""),
                extra={k: v for k, v in d.items() if not k.startswith("_")}))
    return out


# ---- relevance and metrics ---------------------------------------------------------------
def is_relevant(doc_id: str, text: str, gold: Sequence[dict]) -> bool:
    body = canonical_for_match(text)
    return any(g["doc_id"] == doc_id and canonical_for_match(g["span"]) in body for g in gold)


def first_relevant_rank(hits: Sequence[Hit], gold: Sequence[dict]) -> int | None:
    for i, h in enumerate(hits, 1):
        if is_relevant(h.doc_id, h.text, gold):
            return i
    return None


def summarize_ranks(ranks: Sequence[int | None], ks: Sequence[int]) -> dict[str, float]:
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    kmax = max(ks)
    out["mrr"] = sum(1 / r for r in ranks if r is not None and r <= kmax) / n
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean. With a few dozen questions the intervals are wide;
    report them rather than eyeballing point estimates."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ---- retrieval evaluation ------------------------------------------------------------------
def _retrieve(retriever: Retriever, query: str, mode_spec: str, k: int, lang: str | None):
    mode, _, flag = mode_spec.partition("+")
    if flag not in ("", "rerank"):
        raise ValueError(f"Bad mode spec {mode_spec!r}; use dense|sparse|hybrid, optionally '+rerank'")
    if flag == "rerank" and retriever.reranker is None:
        raise ValueError(f"{mode_spec!r} needs a reranker; set reranker.provider in the config.")
    return retriever.retrieve(query, mode=mode, top_k=k, lang=lang, rerank=(flag == "rerank"))


def evaluate_retrieval(retriever: Retriever, questions: Sequence[Question], modes: Sequence[str],
                       ks: Sequence[int] = (1, 3, 5, 10), lang: str | None = None,
                       reference_variant: str | None = None) -> list[dict]:
    answerable = [q for q in questions if q.answerable and q.gold]
    variants = sorted({v for q in answerable for v in q.variants})
    ref = reference_variant if reference_variant in variants else (variants[0] if variants else None)
    kmax = max(ks)
    rows: list[dict] = []
    for mode_spec in modes:
        per_variant_hit5: dict[str, dict[str, float]] = {}
        for variant in variants:
            ranks, lat, qtok, tpw, hit_k = [], [], [], [], {}
            for q in answerable:
                if variant not in q.variants:
                    continue
                text = q.variants[variant]
                hits, tm = _retrieve(retriever, text, mode_spec, kmax, lang)
                rank = first_relevant_rank(hits, q.gold)
                ranks.append(rank)
                hit_k[q.id] = float(rank is not None and rank <= 5)
                lat.append(sum(tm.values()) * 1000)
                nt = retriever.embedder.count_tokens(text)
                qtok.append(nt)
                tpw.append(nt / max(len(text.split()), 1))
            if not ranks:
                continue
            per_variant_hit5[variant] = hit_k
            row = {"mode": mode_spec, "variant": variant, "n": len(ranks),
                   **summarize_ranks(ranks, ks),
                   "hit@5_ci95": bootstrap_ci(list(hit_k.values())),
                   "latency_ms_mean": statistics.mean(lat),
                   "latency_ms_p95": float(np.percentile(lat, 95)),
                   "query_tokens_mean": statistics.mean(qtok),
                   "tokens_per_word": statistics.mean(tpw)}
            rows.append(row)
        # paired delta against the reference variant, on questions that have both
        if ref in per_variant_hit5:
            for row in rows:
                if row["mode"] != mode_spec or row["variant"] == ref:
                    continue
                shared = set(per_variant_hit5[row["variant"]]) & set(per_variant_hit5[ref])
                if shared:
                    row[f"delta_hit@5_vs_{ref}"] = statistics.mean(
                        per_variant_hit5[row["variant"]][i] - per_variant_hit5[ref][i] for i in shared)
    return rows


def format_markdown(rows: Sequence[dict], ks: Sequence[int]) -> str:
    if not rows:
        return "_no rows_"
    delta_cols = sorted({k for r in rows for k in r if k.startswith("delta_")})
    head = ["mode", "variant", "n", *[f"hit@{k}" for k in ks], "MRR", "hit@5 95% CI",
            "tok/word", "p95 ms", *delta_cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lo, hi = r["hit@5_ci95"]
        cells = [r["mode"], r["variant"], str(r["n"]), *[f"{r[f'hit@{k}']:.2f}" for k in ks],
                 f"{r['mrr']:.2f}", f"{lo:.2f}–{hi:.2f}", f"{r['tokens_per_word']:.2f}",
                 f"{r['latency_ms_p95']:.0f}",
                 *[f"{r[c]:+.2f}" if c in r else "" for c in delta_cols]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---- question-set tooling ------------------------------------------------------------------
def validate_questions(questions: Sequence[Question], chunks_by_doc: dict[str, list[str]]) -> list[str]:
    """Return a list of problems (empty list == OK)."""
    problems: list[str] = []
    for qid, c in Counter(q.id for q in questions).items():
        if c > 1:
            problems.append(f"{qid}: duplicate id")
    canon = {d: [canonical_for_match(t) for t in texts] for d, texts in chunks_by_doc.items()}
    for q in questions:
        if not q.variants:
            problems.append(f"{q.id}: no non-empty variants")
        if q.answerable and not q.gold:
            problems.append(f"{q.id}: answerable but has no gold")
        if not q.answerable and q.gold:
            problems.append(f"{q.id}: marked unanswerable but has gold")
        for g in q.gold:
            if g["doc_id"] not in canon:
                problems.append(f"{q.id}: gold doc {g['doc_id']!r} is not in the index")
                continue
            span = canonical_for_match(g["span"])
            if not span:
                problems.append(f"{q.id}: empty gold span")
            elif not any(span in t for t in canon[g["doc_id"]]):
                problems.append(f"{q.id}: span not found inside any single chunk of {g['doc_id']!r} "
                                f"(typo, or it straddles a chunk boundary): {g['span'][:60]!r}")
            elif len(span) > 300:
                problems.append(f"{q.id}: span is long ({len(span)} chars); shorter spans survive "
                                "chunking changes better")
    return problems


def make_template(payloads: Iterable[dict], n: int, n_unanswerable: int, seed: int = 0,
                  min_tokens: int = 40) -> list[dict]:
    """Skeleton question file: sampled passages with pre-filled gold, blank questions for YOU to write."""
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for p in payloads:
        if p.get("n_tokens", 0) >= min_tokens:
            by_lang[p["lang"]].append(p)
    langs = sorted(by_lang)
    rows: list[dict] = []
    per_lang = max(n // max(len(langs), 1), 1)
    i = 0
    for lang in langs:
        for p in rng.sample(by_lang[lang], min(per_lang, len(by_lang[lang]))):
            sents = [s for s in split_sentences(p["text"]) if 30 <= len(s) <= 200] or [p["text"][:200]]
            i += 1
            rows.append({"id": f"q{i:03d}", "group": "",
                         "variants": {"en": "", "ne": "", "ne_rom": "", "ne_typo": ""},
                         "gold": [{"doc_id": p["doc_id"], "span": rng.choice(sents)}],
                         "answerable": True,
                         "_passage_lang": lang, "_passage": p["text"],
                         "_todo": "Write the question in each variant you want to test (delete unused keys). "
                                  "Edit the span so it is the fragment that actually answers it."})
    for _ in range(n_unanswerable):
        i += 1
        rows.append({"id": f"q{i:03d}", "group": "unanswerable",
                     "variants": {"en": "", "ne": "", "ne_rom": "", "ne_typo": ""},
                     "gold": [], "answerable": False,
                     "_todo": "Write a plausible question the corpus does NOT answer."})
    return rows


# ---- generation sheet (hand labeling) ------------------------------------------------------
def build_generation_sheet(pipeline, questions: Sequence[Question], variants: Sequence[str],
                           out_dir: str | Path, mode: str | None = None, seed: int = 0) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full: list[dict] = []
    for q in questions:
        for v in variants:
            if v not in q.variants:
                continue
            a = pipeline.ask(q.variants[v], mode=mode)
            full.append({
                "row_key": len(full), "id": q.id, "variant": v, "answerable": q.answerable,
                "question": q.variants[v], "answer": a.answer, "abstained": a.abstained,
                "cited_docs": [h.doc_id for h in a.citations], "invalid_citations": a.invalid_citations,
                "gold_spans": [g["span"] for g in q.gold], "total_s": a.timings.get("total_s"),
                "query_tokens": a.query_tokens})
    full_path = out / "generation_full.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for r in full:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rows = list(full)
    random.Random(seed).shuffle(rows)  # shuffled, variant hidden: label without knowing the condition
    sheet = out / "generation_labels.csv"
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_key", "question", "gold_spans", "answer", "label", "notes"])
        for r in rows:
            w.writerow([r["row_key"], r["question"], " || ".join(r["gold_spans"]), r["answer"], "", ""])
    return full_path, sheet


def report_labels(full_jsonl: str | Path, labels_csv: str | Path) -> str:
    with open(full_jsonl, encoding="utf-8") as f:
        full = {r["row_key"]: r for r in map(json.loads, (ln for ln in f if ln.strip()))}
    with open(labels_csv, encoding="utf-8-sig", newline="") as f:
        labels = {int(r["row_key"]): r["label"].strip() for r in csv.DictReader(f) if r["label"].strip()}
    table: dict[str, Counter] = defaultdict(Counter)
    for key, label in labels.items():
        table[full[key]["variant"]][label] += 1
    all_labels = sorted({l for c in table.values() for l in c})
    lines = ["| variant | n | " + " | ".join(all_labels) + " |", "|" + "---|" * (len(all_labels) + 2)]
    for v, c in sorted(table.items()):
        n = sum(c.values())
        lines.append(f"| {v} | {n} | " + " | ".join(f"{c[l]} ({100 * c[l] / n:.0f}%)" for l in all_labels) + " |")
    inv = sum(1 for r in full.values() if r["invalid_citations"])
    lines.append(f"\nAnswers citing a non-existent passage: {inv}/{len(full)}")
    return "\n".join(lines)
