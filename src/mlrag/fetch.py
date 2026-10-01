"""Fetch a small multilingual corpus from Wikipedia (CC BY-SA 4.0) via the MediaWiki API.

For each English topic title it also resolves the same article in the other languages through
interlanguage links, so the corpus contains topic-aligned documents (en/ne). That alignment is
what makes cross-lingual retrieval evaluation possible. Nepali articles are often much shorter
than their English counterparts; the coverage report printed at the end makes that visible.

Wikimedia asks API clients to send an identifying User-Agent with contact details, so
``MLRAG_CONTACT`` (an email or URL) is required.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import httpx
import yaml

API = "https://{lang}.wikipedia.org/w/api.php"
LICENSE = "CC BY-SA 4.0"

# Trailing sections that are lists of links/citations, not content. Extend as you inspect the
# data; these are matched case-insensitively against the whole heading.
SKIP_HEADINGS = {
    "en": {"see also", "references", "external links", "further reading", "notes", "sources",
           "bibliography", "citations", "footnotes"},
    "ne": {"सन्दर्भ सामग्रीहरू", "सन्दर्भ सामग्री", "सन्दर्भ", "बाह्य लिङ्कहरू", "बाहिरी कडीहरू",
           "यो पनि हेर्नुहोस्", "थप अध्ययन", "टिप्पणी", "स्रोतहरू"},
}

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")


def parse_extract(text: str, skip: set[str], min_chars: int = 40) -> list[dict]:
    """Split a MediaWiki plain-text extract ('== Heading ==' markers) into sections."""
    skip = {s.casefold() for s in skip}
    sections: list[dict] = []
    heading, lines = "", []
    skipping_level: int | None = None

    def flush() -> None:
        body = "\n".join(l for l in lines if l.strip()).strip()
        if body and len(body) >= min_chars and skipping_level is None:
            sections.append({"heading": heading, "text": body})

    for line in text.split("\n"):
        mt = _HEADING.match(line.strip())
        if mt:
            flush()
            level, name = len(mt.group(1)), mt.group(2).strip()
            lines = []
            if skipping_level is not None and level <= skipping_level:
                skipping_level = None
            if skipping_level is None and name.casefold() in skip:
                skipping_level = level
            heading = name
        else:
            lines.append(line)
    flush()
    return sections


class WikiClient:
    def __init__(self, contact: str, client: httpx.Client | None = None, delay_s: float = 0.3,
                 max_retries: int = 5):
        if not contact:
            raise ValueError("Set MLRAG_CONTACT to an email or URL (Wikimedia User-Agent policy).")
        self.client = client or httpx.Client(
            timeout=30, headers={"User-Agent": f"mlrag-fetch/0.1 ({contact})"})
        self.delay_s, self.max_retries = delay_s, max_retries

    def get_json(self, lang: str, params: dict) -> dict:
        url = API.format(lang=lang)
        for attempt in range(self.max_retries):
            time.sleep(self.delay_s)
            r = self.client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                wait = r.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit() else 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def page(self, lang: str, title: str) -> dict | None:
        data = self.get_json(lang, {
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
            "titles": title, "prop": "extracts|info|revisions|langlinks",
            "explaintext": 1, "exsectionformat": "wiki", "inprop": "url",
            "rvprop": "ids|timestamp", "lllimit": "max"})
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        return pages[0]


def to_doc(page: dict, lang: str, topic: str) -> dict | None:
    sections = parse_extract(page.get("extract", ""), SKIP_HEADINGS.get(lang, set()))
    if not sections:
        return None
    rev = (page.get("revisions") or [{}])[0]
    return {
        "doc_id": f"{lang}:{page['title']}", "topic": topic, "title": page["title"], "lang": lang,
        "url": page.get("fullurl", ""), "revision": rev.get("revid"), "revision_time": rev.get("timestamp"),
        "retrieved": dt.date.today().isoformat(), "license": LICENSE, "source": "wikipedia",
        "sections": sections,
    }


def build_corpus(client: WikiClient, topics: list[str], langs: list[str],
                 source_lang: str = "en") -> tuple[list[dict], list[dict]]:
    docs: list[dict] = []
    report: list[dict] = []
    for topic in topics:
        row = {"topic": topic, **{l: 0 for l in langs}}
        src = client.page(source_lang, topic)
        if src is None:
            print(f"  ! '{topic}' not found on {source_lang}.wikipedia.org")
            report.append(row)
            continue
        links = {l["lang"]: l["title"] for l in src.get("langlinks", [])}
        for lang in langs:
            page = src if lang == source_lang else (
                client.page(lang, links[lang]) if lang in links else None)
            if page is None:
                continue
            doc = to_doc(page, lang, topic)
            if doc:
                docs.append(doc)
                row[lang] = sum(len(s["text"]) for s in doc["sections"])
        report.append(row)
    return docs, report


def run_fetch(topics_file: str, langs: list[str], out: str, contact: str) -> None:
    topics = yaml.safe_load(Path(topics_file).read_text(encoding="utf-8"))["topics"]
    client = WikiClient(contact)
    docs, report = build_corpus(client, topics, langs)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(docs)} documents to {outp}  (text from Wikipedia, {LICENSE}; keep attribution)")
    print(f"{'topic':36s}" + "".join(f"{l:>10s}" for l in langs) + "   (characters of body text)")
    for row in report:
        print(f"{row['topic'][:35]:36s}" + "".join(f"{row[l]:>10d}" for l in langs))
