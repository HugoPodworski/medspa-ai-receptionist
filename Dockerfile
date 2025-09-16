FROM python:3.12-slim

# Do not write .pyc files to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1
# Prefer CPU-only PyTorch wheels
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
# Allow uv to choose best versions across all indexes
ENV UV_INDEX_STRATEGY=unsafe-best-match

# Install uv inside the container (multi-arch friendly)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy lockfile and project config, then install deps
COPY ./uv.lock uv.lock
COPY ./pyproject.toml pyproject.toml
RUN uv sync --no-install-project --no-dev \
    && rm -rf /root/.cache /root/.local/share/uv

# Copy the application code
COPY ./bot.py bot.py
COPY ./model_config.py model_config.py
COPY ./ragprocessing.py ragprocessing.py
COPY ./server.py server.py

# Expose FastAPI port
EXPOSE 7860

# Default command runs via uv to ensure deps/env are applied
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
