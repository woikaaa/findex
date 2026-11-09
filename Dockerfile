# Етап 1: збирання. Тут є uv і все для встановлення залежностей; в готовий образ це не потрапить.
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# спочатку лише залежності за uv.lock: поки lock не змінився, цей шар береться з кешу
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
# потім сам пакет (--no-editable: код копіюється у .venv, а не береться з /app/src)
COPY src ./src
COPY README.md ./
RUN uv sync --locked --no-dev --no-editable

# Етап 2: готовий образ. Лише .venv з етапу 1 і демо-дані, без uv і без вихідників.
FROM python:3.13-slim AS runtime
# звичайний користувач без прав root
RUN useradd --create-home --uid 10001 findex
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --chown=findex:findex demo ./demo
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/demo/index.bin \
    CORPUS_PATH=/app/demo/corpus \
    PORT=8000
USER findex
EXPOSE 8000
# платформа (Render, Docker) питає /health: 503, поки індекс вантажиться, 200, коли готово
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/health', timeout=2)"
CMD ["findex", "serve", "--host", "0.0.0.0"]
