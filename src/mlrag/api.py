"""FastAPI service. Stateless: the client sends the chat history with each request."""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .pipeline import RAGPipeline


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Turn] = []
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    lang: str | None = None  # restrict to a corpus language, e.g. "ne" or "en"


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="mlrag", version="0.1.0",
                  description="Multilingual RAG over Qdrant (hybrid retrieval, cited answers).")
    state: dict[str, RAGPipeline | None] = {"pipeline": pipeline}

    def get_pipeline() -> RAGPipeline:
        if state["pipeline"] is None:  # lazy: the process can start before the index exists
            try:
                state["pipeline"] = RAGPipeline.from_config(load_config())
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=503, detail=f"Pipeline unavailable: {e}") from e
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "pipeline_loaded": state["pipeline"] is not None}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        p = get_pipeline()
        ans = p.ask(req.question, history=[t.model_dump() for t in req.history],
                    mode=req.mode, top_k=req.top_k, lang=req.lang)
        return ans.to_dict()

    @app.post("/search")
    def search(req: AskRequest) -> dict:
        """Retrieval only (no generation): useful for debugging scores and ranking."""
        p = get_pipeline()
        hits, timings = p.retriever.retrieve(req.question, mode=req.mode or p.cfg.retrieval.mode,
                                             top_k=req.top_k or p.cfg.retrieval.top_k, lang=req.lang)
        return {"hits": [h.to_dict(400) for h in hits],
                "timings": {k: round(v, 4) for k, v in timings.items()}}

    return app


app = create_app()  # `uvicorn mlrag.api:app`
