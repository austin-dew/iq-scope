FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npx tsc --noEmit && npx vite build --outDir /web/static --emptyOutDir

FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY backend ./backend
COPY --from=web /web/static ./backend/iqscope/static
RUN pip install --no-cache-dir .

# Runs the test suite on the container's Python: docker build --target test .
# Tests that need particular real recordings skip themselves when those aren't there.
FROM base AS test
RUN pip install --no-cache-dir '.[dev]' && python -m pytest -q

FROM base AS runtime
USER 1000:1000
CMD ["uvicorn", "iqscope.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
