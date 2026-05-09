FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY static /app/static

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

CMD ["openai-router", "--host", "0.0.0.0", "--port", "8000"]
