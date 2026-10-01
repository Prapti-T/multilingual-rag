FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
# `st` pulls in PyTorch (large). For a hosted-embeddings-only image use `.[cohere]` instead
# and set embedding.provider / reranker.provider accordingly.
RUN pip install --no-cache-dir ".[st]"

EXPOSE 8000
CMD ["uvicorn", "mlrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
