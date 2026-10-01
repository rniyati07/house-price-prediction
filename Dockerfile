# syntax=docker/dockerfile:1
# House Price Prediction API image (DOC-04 §11; DOC-05 M12).
#
# Two stages: `builder` installs the locked runtime dependencies (uv.lock, runtime group only)
# and the package; `runtime` adds libgomp1 (SD-14), the non-root user, the venv, src/,
# configs/ and, last, the model artifact (baked in, ADR-15). Build with `make docker-build`,
# which passes MODEL_DIR / MODEL_VERSION and checks the version label against metadata.json:
#
#   docker build --build-arg MODEL_DIR=models/1.0.0 --build-arg MODEL_VERSION=1.0.0 \
#       -t house-price-api:1.0.0 .
ARG PYTHON_IMAGE=python:3.12-slim

# ---------------------------------------------------------------------------- builder
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (lockfile only), so code changes reuse this layer (DOC-04 §11.2).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Then the package itself.
COPY src/ src/
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------- runtime
FROM ${PYTHON_IMAGE} AS runtime
# SD-14: LightGBM links against the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Non-root system user (DOC-04 §11.4): no login shell, no home directory.
RUN useradd --system --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin appuser
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY configs/ /app/configs/

# The model artifact, last: it changes with every release (DOC-04 §11.2, §11.3).
ARG MODEL_DIR
ARG MODEL_VERSION
RUN test -n "${MODEL_DIR}" && test -n "${MODEL_VERSION}" \
    || (echo "build args MODEL_DIR and MODEL_VERSION are required" >&2 && exit 1)
LABEL org.opencontainers.image.title="house-price-api" \
      org.opencontainers.image.version="${MODEL_VERSION}"
COPY ${MODEL_DIR}/model.joblib ${MODEL_DIR}/metadata.json /app/model/

# /app stays root-owned and read-only for appuser; the service never writes to disk.
# HPP_MODEL_DIR=/app/model and HPP_CONFIG_DIR=/app/configs are the settings defaults (SD-08);
# PORT is supplied by Render (default 8000). The model version comes from metadata.json only.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
# Standard-library health check (the slim image has no curl), on the configured PORT.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; url = 'http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health'; sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"]
USER appuser
# SD-09: one Uvicorn worker, no access log; `sh -c` so that Render's PORT is expanded.
CMD ["sh", "-c", "exec uvicorn house_price.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
