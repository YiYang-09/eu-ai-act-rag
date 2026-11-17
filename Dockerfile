# One image for the API, the ingest job and the Streamlit UI (CPU; the LLM runs in Ollama).
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[dense,serve,ui,tracing]"

COPY configs ./configs
COPY scripts ./scripts
COPY data/processed ./data/processed
COPY app ./app
# chunk files are derived from the committed provisions
RUN python scripts/build_corpus.py --chunk-only

EXPOSE 8000 8501
CMD ["uvicorn", "lexrag.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
