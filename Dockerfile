# CPU inference image. The reranker and the retrieval bundle are downloaded from Hugging Face
# at start, at the revisions given by the environment (set by the Kubernetes manifest):
#   docker run -p 8000:8000 -e NEXTTRACK_RERANKER_REVISION=v3 -e NEXTTRACK_RETRIEVAL_REVISION=a_d128_e60 next-track-rec
FROM python:3.12-slim

ARG COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXTTRACK_COMMIT=${COMMIT} \
    NEXTTRACK_RERANKER=N-Mehdi/nexttrack-reranker \
    NEXTTRACK_RETRIEVAL=N-Mehdi/nexttrack-retrieval \
    # Plain HTTP downloads from the Hub: the Xet transfer failed intermittently at start.
    HF_HUB_DISABLE_XET=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app

USER app
EXPOSE 8000
CMD ["uvicorn", "nexttrack.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
