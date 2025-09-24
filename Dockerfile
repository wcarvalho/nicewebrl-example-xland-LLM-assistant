FROM python:3.10-slim

RUN apt update && apt install -y curl procps git build-essential


# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Explicitly copy things in .gitignore
COPY ./config.py /app/

COPY ./google-cloud-key.json /app/

# Copy cached benchmark data to expected location
RUN mkdir -p /root/.xland_minigrid
COPY ./cache/ /root/.xland_minigrid/

# Then copy everything else
COPY . /app

# Install dependencies with uv
RUN rm -rf .venv __pycache__ *.pyc
RUN uv sync --frozen

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

CMD ["uv", "run", "python", "web_app_assistant.py"]