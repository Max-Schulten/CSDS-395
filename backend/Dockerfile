# syntax=docker/dockerfile:1.4

# Stage 1 — dependency installation
#
# A separate stage keeps the pip download cache and build artifacts out of the
# final image layer, even though we end up in a single runtime stage below.
# If you add a separate build-only tool later, this boundary is already in place.

FROM python:3.11-slim AS builder

# System packages required to compile native extensions:
#   gcc / g++       — spaCy Cython extensions, scikit-learn, numpy
#   libgomp1        — OpenMP runtime used by sentence-transformers / torch
#   curl            — used by the health check in the runtime stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the dependency manifest first so this layer is cached as long as
# requirements.txt has not changed — even when source files are edited.
COPY requirements.txt .

# Install CPU-only PyTorch first to avoid pulling the ~2 GB CUDA wheel that
# the default pip index would resolve for torch (required by gliner and
# sentence_transformers).  All other packages are installed from the standard
# index afterward.
RUN pip install --upgrade pip --no-cache-dir \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model so the container starts without an outbound
# network call.  en_core_web_md is what load_nlp() requests by default.
RUN python -m spacy download en_core_web_md


# Stage 2 — runtime image
FROM python:3.11-slim AS runtime

# Runtime-only system libs:
#   libgomp1  — OpenMP, needed at runtime by scikit-learn / torch
#   curl      — used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run ML workloads (or anything else) as root.
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /sbin/nologin appuser

WORKDIR /app

# Create the HuggingFace cache directory owned by appuser so the named volume
# inherits the correct permissions on first mount.
RUN mkdir -p /app/.cache/huggingface && chown -R appuser:appgroup /app/.cache

# Pull the installed Python packages and the spaCy model from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the full repo into the image (build context = repo root).
# models/ and data/ are intentionally included here so the image is self-contained
# when volumes are NOT mounted; docker-compose overrides them with bind-mounts.
# lib/, frontend/, and the root config files must all land under /app so that
# config.py's BASE_DIR (os.path.dirname(os.path.abspath(__file__))) resolves to
# /app, and sys.path.insert(0, "..") in app.py resolves to /app as well.
COPY --chown=appuser:appgroup . /app

# Tell Python to look in /app for top-level imports (config, lib, etc.).
# app.py does `sys.path.insert(0, "..")` at runtime, but setting PYTHONPATH here
# makes the module resolution explicit and independent of cwd.
ENV PYTHONPATH=/app

# Prevent Python from writing .pyc files into the mounted volumes or image layers.
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure stdout/stderr are flushed immediately so Docker logs are not buffered.
ENV PYTHONUNBUFFERED=1

# Flask configuration — host 0.0.0.0 is required so the port is reachable from
# outside the container; debug is off in the container image.
ENV FLASK_DEBUG=0

EXPOSE 5000

# Health check — Flask must return HTTP 200 on / before the service is
# considered healthy.  start_period covers the slow ML model loading phase
# (spaCy + sentence-transformers + GliNER can take 60–120 s on first run).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

USER appuser

# Gunicorn with 1 sync worker — each worker is a separate process with its
# own copy of all models, so there is no shared-state risk between requests.
# A single worker keeps peak memory within the 4 GB compose limit; add a
# second worker only if the memory limit is raised to ~6–8 GB.
# --timeout 120 covers the slow first-request model-warm-up window.
CMD ["gunicorn", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:5000", "backend.app:app"]
