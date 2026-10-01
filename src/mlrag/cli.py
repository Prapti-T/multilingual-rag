# """Command line entry point: ``mlrag <command>``."""
# from __future__ import annotations

# import argparse
# import json
# import os
# import sys
# from datetime import datetime
# from pathlib import Path

# from .config import load_config


# def _store(cfg):
#     from .store import QdrantStore

#     return QdrantStore(cfg.qdrant.collection, cfg.qdrant.url, cfg.qdrant.api_key, cfg.qdrant.path)


# def cmd_fetch(a) -> None:
#     from .fetch import run_fetch

#     run_fetch(a.topics, a.langs, a.out, os.environ.get("MLRAG_CONTACT", ""))


# def cmd_ingest(a) -> None:
#     from .embedders import build_embedder
#     from .ingest import ingest, read_docs

#     cfg = load_config(a.config)
#     stats = ingest(read_docs(a.docs), cfg, build_embedder(cfg.embedding), _store(cfg), recreate=a.recreate)
#     print(json.dumps(stats, indent=2, ensure_ascii=False))


# def cmd_ask(a) -> None:
#     from .pipeline import RAGPipeline

#     cfg = load_config(a.config)
#     ans = RAGPipeline.from_config(cfg).ask(a.question, mode=a.mode, top_k=a.top_k, lang=a.lang)
#     print(ans.answer, "\n")
#     for i, h in enumerate(ans.retrieved, 1):
#         cited = "*" if h in ans.citations else " "
#         print(f"{cited}[{i}] {h.score:.3f}  {h.context} ({h.lang})  {h.url}")
#     print(f"\nmode={ans.mode} query_tokens={ans.query_tokens} timings={ans.timings}")
#     if ans.invalid_citations:
#         print(f"warning: answer cites non-existent passages {ans.invalid_citations}")


# def cmd_serve(a) -> None:
#     import uvicorn

#     uvicorn.run("mlrag.api:app", host=a.host, port=a.port)


# def cmd_template(a) -> None:
#     from .evaluate import make_template

#     cfg = load_config(a.config)
#     rows = make_template(_store(cfg).iter_payloads(), a.n, a.unanswerable, a.seed, a.min_tokens)
#     Path(a.out).parent.mkdir(parents=True, exist_ok=True)
#     with open(a.out, "w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")
#     print(f"Wrote {len(rows)} skeleton questions to {a.out}. Fill in the variants; "
#           "fields starting with '_' are hints and are ignored by the evaluator.")


# def cmd_validate(a) -> None:
#     from collections import defaultdict

#     from .evaluate import load_questions, validate_questions

#     cfg = load_config(a.config)
#     chunks: dict[str, list[str]] = defaultdict(list)
#     for p in _store(cfg).iter_payloads():
#         chunks[p["doc_id"]].append(p["text"])
#     problems = validate_questions(load_questions(a.questions), chunks)
#     print("\n".join(problems) if problems else "OK: all questions and gold spans validate.")
#     sys.exit(1 if problems else 0)


# def cmd_eval(a) -> None:
#     from .evaluate import evaluate_retrieval, format_markdown, load_questions
#     from .pipeline import RAGPipeline

#     cfg = load_config(a.config)
#     retriever = RAGPipeline.from_config(cfg).retriever
#     questions = load_questions(a.questions)
#     rows = evaluate_retrieval(retriever, questions, a.modes, a.ks, lang=a.lang,
#                               reference_variant=a.reference)
#     md = format_markdown(rows, a.ks)
#     print(md)
#     out = Path(a.out)
#     out.mkdir(parents=True, exist_ok=True)
#     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     (out / f"retrieval_{stamp}.md").write_text(md + "\n", encoding="utf-8")
#     meta = {"embedding": vars(cfg.embedding), "chunking": vars(cfg.chunking), "sparse": vars(cfg.sparse),
#             "retrieval": vars(cfg.retrieval), "reranker": vars(cfg.reranker)}
#     (out / f"retrieval_{stamp}.json").write_text(
#         json.dumps({"config": meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
#     print(f"\nSaved to {out}/retrieval_{stamp}.(md|json)")


# def cmd_gen_sheet(a) -> None:
#     from .evaluate import build_generation_sheet, load_questions
#     from .pipeline import RAGPipeline

#     cfg = load_config(a.config)
#     full, sheet = build_generation_sheet(RAGPipeline.from_config(cfg), load_questions(a.questions),
#                                          a.variants, a.out, mode=a.mode)
#     print(f"Label the 'label' column in {sheet} (define your labels first), then run:\n"
#           f"  mlrag gen-report --full {full} --labels {sheet}")


# def cmd_gen_report(a) -> None:
#     from .evaluate import report_labels

#     print(report_labels(a.full, a.labels))


# def main(argv: list[str] | None = None) -> None:
#     for stream in (sys.stdout, sys.stderr):  # Devanagari on Windows consoles
#         if hasattr(stream, "reconfigure"):
#             stream.reconfigure(encoding="utf-8")
#     ap = argparse.ArgumentParser(prog="mlrag", description=__doc__)
#     ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
#     sub = ap.add_subparsers(dest="cmd", required=True)

#     p = sub.add_parser("fetch", help="Download a topic-aligned Wikipedia corpus")
#     p.add_argument("--topics", default="data/topics.yaml")
#     p.add_argument("--langs", nargs="+", default=["en", "ne"])
#     p.add_argument("--out", default="data/raw/wiki_docs.jsonl")
#     p.set_defaults(fn=cmd_fetch)

#     p = sub.add_parser("ingest", help="Chunk, embed and index documents into Qdrant")
#     p.add_argument("--docs", default="data/raw/wiki_docs.jsonl")
#     p.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
#     p.set_defaults(fn=cmd_ingest)

#     p = sub.add_parser("ask", help="Ask one question")
#     p.add_argument("question")
#     p.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
#     p.add_argument("--top-k", type=int)
#     p.add_argument("--lang")
#     p.set_defaults(fn=cmd_ask)

#     p = sub.add_parser("serve", help="Run the FastAPI service")
#     p.add_argument("--host", default="127.0.0.1")
#     p.add_argument("--port", type=int, default=8000)
#     p.set_defaults(fn=cmd_serve)

#     p = sub.add_parser("template", help="Create a skeleton question file from the indexed corpus")
#     p.add_argument("--n", type=int, default=40)
#     p.add_argument("--unanswerable", type=int, default=8)
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--min-tokens", type=int, default=40, help="skip chunks shorter than this")
#     p.add_argument("--out", default="eval/questions.todo.jsonl")
#     p.set_defaults(fn=cmd_template)

#     p = sub.add_parser("validate", help="Check gold spans against the indexed chunks")
#     p.add_argument("--questions", default="eval/questions.jsonl")
#     p.set_defaults(fn=cmd_validate)

#     p = sub.add_parser("eval", help="Retrieval evaluation: hit@k / MRR per mode and query variant")
#     p.add_argument("--questions", default="eval/questions.jsonl")
#     p.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid"],
#                    help="dense | sparse | hybrid, optionally suffixed +rerank")
#     p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
#     p.add_argument("--lang", help="restrict search to one corpus language")
#     p.add_argument("--reference", default="ne", help="variant that deltas are computed against")
#     p.add_argument("--out", default="results")
#     p.set_defaults(fn=cmd_eval)

#     p = sub.add_parser("gen-sheet", help="Generate answers and write a shuffled sheet for hand labeling")
#     p.add_argument("--questions", default="eval/questions.jsonl")
#     p.add_argument("--variants", nargs="+", default=["en", "ne", "ne_rom"])
#     p.add_argument("--mode")
#     p.add_argument("--out", default="results")
#     p.set_defaults(fn=cmd_gen_sheet)

#     p = sub.add_parser("gen-report", help="Summarise hand labels from a gen-sheet")
#     p.add_argument("--full", default="results/generation_full.jsonl")
#     p.add_argument("--labels", default="results/generation_labels.csv")
#     p.set_defaults(fn=cmd_gen_report)

#     args = ap.parse_args(argv)
#     args.fn(args)


# if __name__ == "__main__":
#     main()

"""Command line entry point: ``mlrag <command>``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config, load_env_file


def _store(cfg):
    from .store import QdrantStore

    return QdrantStore(cfg.qdrant.collection, cfg.qdrant.url, cfg.qdrant.api_key, cfg.qdrant.path)


def cmd_fetch(a) -> None:
    from .fetch import run_fetch

    load_env_file()  # fetch doesn't call load_config, so read MLRAG_CONTACT from .env here
    run_fetch(a.topics, a.langs, a.out, os.environ.get("MLRAG_CONTACT", ""))


def cmd_ingest(a) -> None:
    from .embedders import build_embedder
    from .ingest import ingest, read_docs

    cfg = load_config(a.config)
    stats = ingest(read_docs(a.docs), cfg, build_embedder(cfg.embedding), _store(cfg), recreate=a.recreate)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def cmd_ask(a) -> None:
    from .pipeline import RAGPipeline

    cfg = load_config(a.config)
    ans = RAGPipeline.from_config(cfg).ask(a.question, mode=a.mode, top_k=a.top_k, lang=a.lang)
    print(ans.answer, "\n")
    for i, h in enumerate(ans.retrieved, 1):
        cited = "*" if h in ans.citations else " "
        print(f"{cited}[{i}] {h.score:.3f}  {h.context} ({h.lang})  {h.url}")
    print(f"\nmode={ans.mode} query_tokens={ans.query_tokens} timings={ans.timings}")
    if ans.invalid_citations:
        print(f"warning: answer cites non-existent passages {ans.invalid_citations}")


def cmd_serve(a) -> None:
    import uvicorn

    uvicorn.run("mlrag.api:app", host=a.host, port=a.port)


def cmd_template(a) -> None:
    from .evaluate import make_template

    cfg = load_config(a.config)
    rows = make_template(_store(cfg).iter_payloads(), a.n, a.unanswerable, a.seed, a.min_tokens)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} skeleton questions to {a.out}. Fill in the variants; "
          "fields starting with '_' are hints and are ignored by the evaluator.")


def cmd_validate(a) -> None:
    from collections import defaultdict

    from .evaluate import load_questions, validate_questions

    cfg = load_config(a.config)
    chunks: dict[str, list[str]] = defaultdict(list)
    for p in _store(cfg).iter_payloads():
        chunks[p["doc_id"]].append(p["text"])
    problems = validate_questions(load_questions(a.questions), chunks)
    print("\n".join(problems) if problems else "OK: all questions and gold spans validate.")
    sys.exit(1 if problems else 0)


def cmd_eval(a) -> None:
    from .evaluate import evaluate_retrieval, format_markdown, load_questions
    from .pipeline import RAGPipeline

    cfg = load_config(a.config)
    retriever = RAGPipeline.from_config(cfg).retriever
    questions = load_questions(a.questions)
    rows = evaluate_retrieval(retriever, questions, a.modes, a.ks, lang=a.lang,
                              reference_variant=a.reference)
    md = format_markdown(rows, a.ks)
    print(md)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"retrieval_{stamp}.md").write_text(md + "\n", encoding="utf-8")
    meta = {"embedding": vars(cfg.embedding), "chunking": vars(cfg.chunking), "sparse": vars(cfg.sparse),
            "retrieval": vars(cfg.retrieval), "reranker": vars(cfg.reranker)}
    (out / f"retrieval_{stamp}.json").write_text(
        json.dumps({"config": meta, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}/retrieval_{stamp}.(md|json)")


def cmd_gen_sheet(a) -> None:
    from .evaluate import build_generation_sheet, load_questions
    from .pipeline import RAGPipeline

    cfg = load_config(a.config)
    full, sheet = build_generation_sheet(RAGPipeline.from_config(cfg), load_questions(a.questions),
                                         a.variants, a.out, mode=a.mode)
    print(f"Label the 'label' column in {sheet} (define your labels first), then run:\n"
          f"  mlrag gen-report --full {full} --labels {sheet}")


def cmd_gen_report(a) -> None:
    from .evaluate import report_labels

    print(report_labels(a.full, a.labels))


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # Devanagari on Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="mlrag", description=__doc__)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="Download a topic-aligned Wikipedia corpus")
    p.add_argument("--topics", default="data/topics.yaml")
    p.add_argument("--langs", nargs="+", default=["en", "ne"])
    p.add_argument("--out", default="data/raw/wiki_docs.jsonl")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("ingest", help="Chunk, embed and index documents into Qdrant")
    p.add_argument("--docs", default="data/raw/wiki_docs.jsonl")
    p.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="Ask one question")
    p.add_argument("question")
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--lang")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("serve", help="Run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("template", help="Create a skeleton question file from the indexed corpus")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--unanswerable", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-tokens", type=int, default=40, help="skip chunks shorter than this")
    p.add_argument("--out", default="eval/questions.todo.jsonl")
    p.set_defaults(fn=cmd_template)

    p = sub.add_parser("validate", help="Check gold spans against the indexed chunks")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("eval", help="Retrieval evaluation: hit@k / MRR per mode and query variant")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--modes", nargs="+", default=["dense", "sparse", "hybrid"],
                   help="dense | sparse | hybrid, optionally suffixed +rerank")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--lang", help="restrict search to one corpus language")
    p.add_argument("--reference", default="ne", help="variant that deltas are computed against")
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("gen-sheet", help="Generate answers and write a shuffled sheet for hand labeling")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--variants", nargs="+", default=["en", "ne", "ne_rom"])
    p.add_argument("--mode")
    p.add_argument("--out", default="results")
    p.set_defaults(fn=cmd_gen_sheet)

    p = sub.add_parser("gen-report", help="Summarise hand labels from a gen-sheet")
    p.add_argument("--full", default="results/generation_full.jsonl")
    p.add_argument("--labels", default="results/generation_labels.csv")
    p.set_defaults(fn=cmd_gen_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()