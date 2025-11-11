FROM ghcr.io/astral-sh/uv:python3.10

WORKDIR /app

COPY pyproject.toml ./

# Preload dependencies to leverage Docker layer caching
RUN uv sync --no-install-project

COPY . .

RUN uv sync

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
