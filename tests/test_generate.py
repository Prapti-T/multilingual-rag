import json

import httpx

from mlrag.generate import (ABSTAIN_TEXT, NO_ANSWER, ExtractiveGenerator, OllamaGenerator,
                            OpenAICompatGenerator, build_context, build_messages, parse_generation)
from mlrag.types import Hit


def h(i, text="passage text"):
    return Hit(id=str(i), score=1.0, text=text, context=f"Doc{i} > Sec", doc_id=f"d{i}", title=f"Doc{i}",
               lang="en", url="", chunk_index=i)


def test_parse_valid_and_invalid_citations():
    g = parse_generation("Everest is highest [1]. Also [2] and [1] and [7].", n_passages=2)
    assert g.cited == [1, 2] and g.invalid == [7] and not g.abstained


def test_sentinel_means_abstention_with_fixed_text():
    g = parse_generation(f"  {NO_ANSWER} ", n_passages=3)
    assert g.abstained and g.text == ABSTAIN_TEXT and g.cited == []


def test_think_blocks_are_stripped():
    g = parse_generation("<think>hidden reasoning [9]</think>The answer [1].", n_passages=1)
    assert g.text == "The answer [1]." and g.cited == [1] and g.invalid == []


def test_build_context_numbers_passages_and_respects_budget():
    ctx = build_context([h(1, "a" * 50), h(2, "b" * 50), h(3, "c" * 50)], max_chars=160)  # ~71 chars each
    assert "[1] (Doc1 > Sec; en)" in ctx and "[2]" in ctx and "[3]" not in ctx
    assert "[1]" in build_context([h(1, "x" * 500)], max_chars=10)  # always keeps at least one


def test_build_messages_trims_history_and_puts_passages_in_last_turn():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    msgs = build_messages("Q?", [h(1)], history, max_context_chars=1000, history_turns=2)
    assert msgs[0]["role"] == "system" and NO_ANSWER in msgs[0]["content"]
    assert [m["content"] for m in msgs[1:-1]] == ["m6", "m7", "m8", "m9"]
    assert msgs[-1]["role"] == "user" and "[1]" in msgs[-1]["content"] and "Q?" in msgs[-1]["content"]


def test_extractive_generator():
    g = ExtractiveGenerator(max_chars=10).generate("q", [h(1, "x" * 50)], None)
    assert g.cited == [1] and g.text.endswith("… [1]")
    assert ExtractiveGenerator().generate("q", [], None).abstained


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_request_shape_and_parsing():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "Kathmandu [1]"}})

    gen = OllamaGenerator("qwen2.5:3b", "http://ollama.test:11434/", 0.0, 1000, 3, 5, client=_client(handler))
    out = gen.generate("Capital?", [h(1)], None)
    assert seen["url"] == "http://ollama.test:11434/api/chat"
    assert seen["body"]["model"] == "qwen2.5:3b" and seen["body"]["stream"] is False
    assert seen["body"]["options"]["temperature"] == 0.0
    assert out.text == "Kathmandu [1]" and out.cited == [1]


def test_openai_compat_request_shape_and_auth_header():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": NO_ANSWER}}]})

    gen = OpenAICompatGenerator("some-model", "https://llm.test/v1", 0.0, 1000, 3, 5,
                                api_key="k", client=_client(handler))
    out = gen.generate("q", [h(1)], None)
    assert seen["url"] == "https://llm.test/v1/chat/completions" and seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == "some-model" and out.abstained
