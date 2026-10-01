import json

import pytest
from fastapi.testclient import TestClient

from mlrag.api import create_app
from mlrag.cli import main
from mlrag.config import load_config


# ---- API ----------------------------------------------------------------------------------
def test_api_health_ask_search(index):
    client = TestClient(create_app(index.pipeline))
    assert client.get("/health").json() == {"status": "ok", "pipeline_loaded": True}

    r = client.post("/ask", json={"question": "नेपालको राजधानी के हो?", "top_k": 3, "mode": "hybrid"})
    body = r.json()
    assert r.status_code == 200
    assert body["retrieved"][0]["doc_id"] == "ne:काठमाडौं"
    assert body["citations"][0]["doc_id"] == "ne:काठमाडौं" and body["abstained"] is False
    assert body["query_tokens"] > 0 and "total_s" in body["timings"]

    r = client.post("/search", json={"question": "highest mountain", "lang": "en", "top_k": 2})
    assert r.status_code == 200 and all(h["lang"] == "en" for h in r.json()["hits"])


def test_api_validates_input(index):
    client = TestClient(create_app(index.pipeline))
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422


def test_api_reports_503_when_no_index_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLRAG_CONFIG", raising=False)
    r = TestClient(create_app()).post("/ask", json={"question": "x"})
    assert r.status_code == 503 and "mlrag ingest" in r.json()["detail"]


# ---- config -------------------------------------------------------------------------------
def test_config_yaml_overrides_defaults_and_env_overrides_secrets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the repo's own .env out of the test
    p = tmp_path / "c.yaml"
    p.write_text("retrieval:\n  mode: dense\n  top_k: 7\nqdrant:\n  collection: mine\n", encoding="utf-8")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("LLM_MODEL", "tiny")
    cfg = load_config(str(p))
    assert (cfg.retrieval.mode, cfg.retrieval.top_k, cfg.qdrant.collection) == ("dense", 7, "mine")
    assert cfg.qdrant.url == "http://qdrant.test:6333" and cfg.generation.model == "tiny"
    assert cfg.chunking.max_tokens == 256  # untouched default


def test_config_rejects_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "c.yaml"
    p.write_text("retrieval:\n  topk: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="topk"):
        load_config(str(p))


def test_every_shipped_config_loads(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    configs = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert len(configs) >= 3
    loaded = {p.stem: load_config(str(p)) for p in configs}
    assert loaded["default"].embedding.model == "BAAI/bge-m3" and loaded["default"].retrieval.mode == "hybrid"
    assert loaded["e5-small"].embedding.doc_prefix == "passage: "
    assert loaded["cohere"].reranker.provider == "cohere"
    # each alternative gets its own collection so indexes never collide
    assert len({c.qdrant.collection for c in loaded.values()}) == len(loaded)


# ---- CLI, end to end on an embedded (on-disk) Qdrant with the offline embedder ----------------
def test_cli_pipeline_ingest_ask_template_validate_eval(tmp_path, monkeypatch, capsys):
    from tests.conftest import DOCS

    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs.jsonl"
    docs.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in DOCS), encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"qdrant:\n  path: {tmp_path / 'qd'}\nembedding:\n  provider: hash\n"
                   f"index:\n  meta_dir: {tmp_path / 'idx'}\n", encoding="utf-8")
    base = ["--config", str(cfg)]

    main(base + ["ingest", "--docs", str(docs), "--recreate"])
    assert json.loads(capsys.readouterr().out)["chunks"] == 5

    main(base + ["ask", "नेपालको राजधानी के हो?", "--top-k", "2"])
    out = capsys.readouterr().out
    assert "काठमाडौं" in out and "query_tokens=" in out

    todo = tmp_path / "todo.jsonl"
    main(base + ["template", "--n", "4", "--unanswerable", "1", "--min-tokens", "5", "--out", str(todo)])
    capsys.readouterr()
    assert len(todo.read_text(encoding="utf-8").splitlines()) == 5

    qs = tmp_path / "questions.jsonl"
    qs.write_text(json.dumps({"id": "q1", "variants": {"ne": "नेपालको राजधानी के हो?"},
                              "gold": [{"doc_id": "ne:काठमाडौं", "span": "नेपालको राजधानी हो"}]},
                             ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(base + ["validate", "--questions", str(qs)])
    assert e.value.code == 0 and "OK" in capsys.readouterr().out

    main(base + ["eval", "--questions", str(qs), "--modes", "dense", "sparse", "hybrid",
                 "--out", str(tmp_path / "res")])
    assert "hit@5 95% CI" in capsys.readouterr().out
    assert list((tmp_path / "res").glob("retrieval_*.json")) and list((tmp_path / "res").glob("retrieval_*.md"))
