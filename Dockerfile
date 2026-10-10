FROM ghcr.io/astral-sh/uv:0.12.22-python3.11-trixie-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy DATA_DIR=/app/data

# Install deps first so this layer is cached when only code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
