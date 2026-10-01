import httpx
import pytest

from mlrag.fetch import SKIP_HEADINGS, WikiClient, build_corpus, parse_extract

EN_EXTRACT = (
    "Lead paragraph about photosynthesis that is long enough to pass the filter.\n\n\n"
    "== Process ==\nPlants use light. This text is long enough to be kept as a section.\n"
    "=== Details ===\nMore details here that are definitely long enough to keep around.\n"
    "== References ==\nSome citation text that is long enough to be skipped entirely.\n"
    "=== Sub reference ===\nNested text under references that must also be skipped entirely.\n"
    "== Impact ==\nImpact paragraph that is long enough to remain after filtering."
)
NE_EXTRACT = (
    "प्रकाश संश्लेषणमा बिरुवाले सूर्यको प्रकाश प्रयोग गरेर खाना बनाउँछ। यो परिचय खण्ड हो।\n\n"
    "== प्रक्रिया ==\nयो प्रक्रिया खण्डमा पर्याप्त लामो पाठ छ ताकि यसलाई राखियोस्।\n"
    "== सन्दर्भ सामग्रीहरू ==\nयहाँ सन्दर्भहरूको सूची छ जसलाई पूर्ण रूपमा हटाउनुपर्छ।"
)


def page(title, extract, links=()):
    return {"query": {"pages": [{
        "pageid": 1, "title": title, "extract": extract,
        "fullurl": f"https://example.test/{title}",
        "revisions": [{"revid": 111, "timestamp": "2026-01-01T00:00:00Z"}],
        "langlinks": [{"lang": l, "title": t} for l, t in links]}]}}


def make_client(handler, **kw):
    transport = httpx.MockTransport(handler)
    return WikiClient("tester@example.com", httpx.Client(transport=transport,
                      headers={"User-Agent": "mlrag-fetch/0.1 (tester@example.com)"}), delay_s=0, **kw)


def test_parse_extract_skips_reference_sections_and_their_children():
    sections = parse_extract(EN_EXTRACT, SKIP_HEADINGS["en"])
    assert [s["heading"] for s in sections] == ["", "Process", "Details", "Impact"]
    assert all("citation" not in s["text"].lower() for s in sections)


def test_parse_extract_handles_nepali_reference_heading():
    sections = parse_extract(NE_EXTRACT, SKIP_HEADINGS["ne"])
    assert [s["heading"] for s in sections] == ["", "प्रक्रिया"]


def test_build_corpus_aligns_languages_and_reports_missing_topics():
    def handler(request: httpx.Request) -> httpx.Response:
        host, title = request.url.host, request.url.params["titles"]
        assert "mlrag-fetch" in request.headers["User-Agent"]
        if host == "en.wikipedia.org" and title == "Photosynthesis":
            return httpx.Response(200, json=page("Photosynthesis", EN_EXTRACT,
                                                 [("ne", "प्रकाश संश्लेषण"), ("de", "Photosynthese")]))
        if host == "ne.wikipedia.org" and title == "प्रकाश संश्लेषण":
            return httpx.Response(200, json=page("प्रकाश संश्लेषण", NE_EXTRACT))
        return httpx.Response(200, json={"query": {"pages": [{"title": title, "missing": True}]}})

    docs, report = build_corpus(make_client(handler), ["Photosynthesis", "Nonexistent"], ["en", "ne"])
    assert [d["doc_id"] for d in docs] == ["en:Photosynthesis", "ne:प्रकाश संश्लेषण"]
    assert all(d["topic"] == "Photosynthesis" and d["license"] == "CC BY-SA 4.0" for d in docs)
    assert docs[0]["revision"] == 111 and docs[0]["url"].startswith("https://")
    assert report[0]["en"] > 0 and report[0]["ne"] > 0
    assert report[1] == {"topic": "Nonexistent", "en": 0, "ne": 0}


def test_topic_without_target_language_article_is_kept_for_source_language_only():
    def handler(request):
        return httpx.Response(200, json=page("Gravity", EN_EXTRACT, [("de", "Gravitation")]))

    docs, report = build_corpus(make_client(handler), ["Gravity"], ["en", "ne"])
    assert [d["lang"] for d in docs] == ["en"] and report[0]["ne"] == 0


def test_retries_on_rate_limit_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"query": {"pages": []}})

    assert make_client(handler).get_json("en", {"action": "query"}) == {"query": {"pages": []}}
    assert calls["n"] == 2


def test_contact_is_required():
    with pytest.raises(ValueError, match="MLRAG_CONTACT"):
        WikiClient("")
