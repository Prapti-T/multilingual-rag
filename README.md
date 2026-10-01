# mlrag: multilingual (Nepali / English) RAG on Qdrant

A small, readable RAG system for question answering over a mixed Nepali–English corpus, built
around the things that go wrong when a retrieval pipeline built for English meets Devanagari,
Romanized Nepali and inconsistent spelling. Retrieval quality is *measured*, not assumed: the
repo ships an evaluation harness that compares dense, sparse and hybrid retrieval across query
variants (Devanagari, Romanized, misspelled, English).

**Provenance.** This is an independent implementation on public data (Wikipedia, CC BY-SA 4.0). It
was written to demonstrate the retrieval and evaluation approach I use for multilingual educational
RAG; it contains no code, data, prompts or architecture from any employer.

```
Wikipedia (en + aligned ne)                                   query (ne / Romanized ne / en)
        │  mlrag fetch                                               │
        ▼                                                            ▼
 sections ──► sentence-aware chunks ──► dense vector (any embedder)   dense query vector ─┐
              (token budget from        + BM25 sparse vector           BM25 query vector ─┤
               the embedder's own       (Unicode-aware analyzer)                           ▼
               tokenizer)                        │                     Qdrant: prefetch both, fuse with RRF
                                                 ▼                                         │
                                  Qdrant collection (named dense + sparse vectors)          ▼
                                                                          optional cross-encoder / Cohere rerank
                                                                                           │
                                                                                           ▼
                                                              grounded answer with [n] citations, or [NO_ANSWER]
```

## What is specific to multilingual retrieval here

| Problem | What the code does | Where |
|---|---|---|
| Python's `\w+` splits Devanagari words at every vowel sign (matras are combining marks, not "alphanumeric") | Tokenizer walks Unicode categories (`L`, `M`, `N`) so *नेपालको* stays one token | `text.py`, tested in `test_text.py` |
| Hrasva/dirgha (short/long vowel) spelling varies in real text, so lexical match silently fails | Optional vowel-length folding in the **sparse analyzer only** (display text is never altered); evaluate on/off with `ne_typo` queries | `text.py`, `sparse.py` |
| Nepali sentences end with `।`, not only `.` | Sentence splitter handles danda (`।`, `॥`) | `text.split_sentences` |
| Chunking with one tokenizer and embedding with another overflows or wastes context, and Devanagari costs many more tokens per word than English | Chunk budget is measured with the **embedder's own tokenizer**; `tok/word` is reported per query variant in the evaluation | `chunking.py`, `embedders.py` |
| Dense and BM25 scores live on different scales, so thresholding or averaging them is fragile | Reciprocal Rank Fusion, server-side in Qdrant, no score calibration; reranker is optional and swappable | `store.py`, `retrieve.py` |
| Corpus changes shouldn't need a vocabulary rebuild | Terms hashed to u32 sparse indices; IDF applied by Qdrant (`Modifier.IDF`) so it stays correct as the corpus grows | `sparse.py` |
| Re-ingesting duplicates data | Deterministic chunk ids (uuid5 of document + position) make ingestion idempotent | `chunking.py` |
| Chunk ids change whenever chunking changes, breaking an eval set | Gold labels are `(doc_id, verbatim span)`; a chunk is relevant if it comes from that document and contains the span | `evaluate.py` |
| Wrong-language or unsupported answers | Prompt requires answering in the question's language/script and citing passages; `[NO_ANSWER]` sentinel gives a measurable abstention signal; citations pointing at non-existent passages are flagged | `generate.py` |

## Quickstart

Requires Python 3.10+. The default embedder (`BAAI/bge-m3`) is a ~570M-parameter download and is
slow on CPU; see *Choosing models* below for smaller options.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[st,dev]"                              # st = sentence-transformers (local models)
cp .env.example .env                                    # set MLRAG_CONTACT (Wikimedia User-Agent policy)

mlrag fetch                    # topics in data/topics.yaml -> data/raw/wiki_docs.jsonl (+ coverage table)
mlrag ingest --recreate        # chunk, embed, index into embedded Qdrant (./data/qdrant_local)
mlrag ask "नेपालको राजधानी कहाँ हो?"
mlrag serve                    # FastAPI on :8000 (docs at /docs)
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Where is Lumbini?", "mode": "hybrid", "top_k": 5}'
```

**Qdrant server instead of embedded mode** (embedded mode locks its folder to one process, so the
API and the CLI cannot run at the same time):

```bash
docker compose up -d qdrant
echo 'QDRANT_URL=http://localhost:6333' >> .env
```

**Answer generation.** The default `generation.provider: extractive` returns the best passage with a
citation and needs no LLM, so retrieval can be evaluated for free. For generated answers set
`provider: ollama` (local, `ollama pull <model>` first) or `provider: openai-compat` (any
OpenAI-compatible endpoint; set `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` in `.env`).

## Choosing models (compare them, don't assume)

Nothing here is tuned to one model. Each configuration gets its own Qdrant collection, and the
index metadata records which embedder built it (a mismatch is refused at startup).

```bash
MLRAG_CONFIG=configs/e5-small.yaml mlrag ingest --recreate     # lighter, CPU-friendly
MLRAG_CONFIG=configs/cohere.yaml   mlrag ingest --recreate     # hosted; needs COHERE_API_KEY
```

Check each model card for Nepali/Devanagari coverage and parameter count, and confirm current hosted
model names in the provider's docs; the names in `configs/` are starting points.

## Evaluating retrieval

The question file is yours to write; the tooling makes that fast and keeps it honest.

```bash
mlrag template --n 40 --unanswerable 8     # samples indexed passages -> eval/questions.todo.jsonl
# For each item: write the question in the variants you want to test, and edit "span"
# to the fragment that actually answers it. Save as eval/questions.jsonl.
mlrag validate                             # every gold span must exist inside a single indexed chunk
mlrag eval --modes dense sparse hybrid hybrid+rerank --reference ne
```

```json
{"id": "q007", "group": "geography",
 "variants": {"en": "...", "ne": "...", "ne_rom": "...", "ne_typo": "..."},
 "gold": [{"doc_id": "ne:<title>", "span": "short verbatim fragment"}],
 "answerable": true}
```

`variants` are free-form keys, one row per (mode, variant) in the output. Suggested conventions:
`ne` Devanagari, `ne_rom` Romanized Nepali (ideally written by several people, since Romanization has
no standard spelling), `ne_typo` realistic hrasva/dirgha errors, `en` English. List more than one
`gold` entry (for example the Nepali and English passage) to credit cross-lingual matches.

The report contains hit@k, MRR, a bootstrap 95% CI for hit@5 (with a few dozen questions the intervals
are wide, so read them before reading differences), mean tokens per word and p95 latency per variant,
and a paired Δhit@5 against the reference variant. Outputs go to `results/retrieval_<timestamp>.{md,json}`
together with the configuration that produced them.

**Generation**, evaluated by hand (an LLM judge would add its own multilingual blind spots):

```bash
mlrag gen-sheet --variants en ne ne_rom    # answers + shuffled CSV with the variant column hidden
# fill the `label` column (define your labels before you start), then:
mlrag gen-report
```

Unanswerable questions measure abstention. The script of the question still reveals the variant while
labeling, so blinding is partial.

## Results

_Not filled in yet: this repository ships the harness, and numbers depend on the corpus, the question
set and the models you pick. Run `mlrag eval` and paste the table here, together with the model,
chunking and sparse settings it printed, the number of questions, and what the failures looked like._

## Known limitations

- **Romanized Nepali is not bridged lexically.** The sparse branch cannot match *rajdhani* to *राजधानी*;
  that case relies on the dense model. A transliteration or query-normalisation step is the obvious next
  experiment and `mlrag eval` is how you would show whether it helps.
- **Spelling folding is deliberately small** (vowel length, chandrabindu/anusvara). Other common confusions
  (श/ष/स, ब/व) are not folded.
- **Multi-turn chat is stateless and shallow**: history is passed to the generator but retrieval uses only
  the current question; there is no query rewriting.
- **Corpus asymmetry.** Nepali Wikipedia articles are often far shorter than their English counterparts, which
  affects cross-lingual comparisons. `mlrag fetch` prints per-language coverage.
- **Section filters for Nepali** (`fetch.SKIP_HEADINGS`) are a short hand-written list; inspect the data and extend it.
- **Test coverage is offline.** The test suite (`pytest`) exercises chunking, the analyzer, Qdrant hybrid
  search, evaluation, the API and the CLI end to end using an in-memory Qdrant and a deterministic, non-semantic
  hash embedder, plus mocked HTTP for Wikipedia, Ollama and OpenAI-compatible endpoints. It does not download
  models or call hosted services, so retrieval *quality* is only established by running `mlrag eval` on real data.
  The local sentence-transformers/cross-encoder adapters, the Docker files and the Streamlit UI are thin and have
  not been exercised by the tests.

## Layout

```
src/mlrag/
  text.py          Unicode normalisation, tokenizer, sentence splitter, spelling folding
  chunking.py      sentence-aware chunks with a token budget
  sparse.py        BM25 sparse encoder (hashed terms, Qdrant-side IDF)
  embedders.py     sentence-transformers | cohere | hash (tests only)
  rerankers.py     cross-encoder | cohere | none
  store.py         Qdrant: dense + sparse named vectors, RRF hybrid search, filters
  ingest.py        documents -> chunks -> vectors -> Qdrant (+ index metadata)
  retrieve.py      dense / sparse / hybrid [+ rerank] with timings
  generate.py      grounded prompt, citations, abstention; extractive | ollama | openai-compat
  pipeline.py      retrieve -> generate -> Answer
  evaluate.py      metrics, bootstrap CIs, question validation, templates, generation sheets
  fetch.py         topic-aligned Wikipedia corpus
  api.py, cli.py   FastAPI service and `mlrag` command
app/ui.py          optional Streamlit front end
examples/mlm_demo.py   optional: tokenizer fragmentation and fill-mask with a multilingual MLM
tests/             offline test suite
```

## Data and licence

Wikipedia text is CC BY-SA 4.0. `mlrag fetch` records each article's URL, revision id and retrieval date;
keep that attribution if you publish derived data. `data/raw/` is git-ignored on purpose.
