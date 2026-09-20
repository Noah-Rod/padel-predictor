# Single image, three entrypoints: the pipelines, the API and the UI all run the
# same code. Dependencies come from uv.lock, so the image is reproducible.
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the project itself.
COPY README.md ./
COPY src/ src/
COPY config/ config/
COPY ui/ ui/
RUN uv sync --frozen --no-dev

# Run as a non-root user.
RUN useradd --create-home --uid 10001 padel \
    && mkdir -p /app/data \
    && chown -R padel:padel /app
USER padel

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,os,sys; \
sys.exit(0 if httpx.get(f\"http://localhost:{os.getenv('PADEL_API_PORT','8000')}/health\", timeout=4).status_code==200 else 1)"

# Default: the prediction API. Override the command to run a pipeline or the UI:
#   docker run --rm padel-predictor padel feature-pipeline
#   docker run --rm padel-predictor streamlit run ui/app.py
CMD ["padel", "serve"]
