# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy all files needed for build (including README for hatchling)
COPY . .

# Build with uv (compile bytecode and lockfile)
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -s /bin/bash taskgate

# Copy venv from builder
COPY --from=builder --chown=taskgate:taskgate /app/.venv /app/.venv

# Copy application code
COPY --chown=taskgate:taskgate . .

# Set PATH to use venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER taskgate

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import taskgate; print('ok')" || exit 1

ENTRYPOINT ["taskgate"]
CMD ["--help"]
