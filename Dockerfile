# CPU inference image. Artifacts and audio embeddings are mounted at run time:
#   docker run -p 8000:8000 -v "$PWD/artifacts:/artifacts:ro" -v "$PWD/data/audio:/audio:ro" next-track-rec
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXTTRACK_ARTIFACTS=/artifacts \
    NEXTTRACK_AUDIO=/audio

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" && useradd --create-home app

USER app
EXPOSE 8000
CMD ["uvicorn", "nexttrack.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
