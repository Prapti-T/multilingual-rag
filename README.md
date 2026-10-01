# mlrag: multilingual (Nepali / English) RAG on Qdrant

A small, readable RAG system for question answering over a mixed Nepali–English corpus, built
around the things that go wrong when a retrieval pipeline built for English meets Devanagari,
Romanized Nepali and inconsistent spelling. Retrieval quality is *measured*, not assumed: the
repo ships an evaluation harness that compares dense, sparse and hybrid retrieval across query
variants (Devanagari, Romanized, misspelled, English).

**Provenance.** This is an independent implementation on public data (Wikipedia, CC BY-SA 4.0). It
was written to demonstrate the retrieval and evaluation approach I use for multilingual educational
RAG; it contains no code, data, prompts or architecture from any employer.

