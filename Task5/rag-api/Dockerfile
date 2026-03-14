FROM python:3.11-slim

# Чтобы sentence-transformers/torch нормально ставились
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY app_factory.py /app/
COPY api/ /app/api/
COPY rag/ /app/rag/

# Папка для базы Chroma
# В docker-compose смонтируем сюда volume
RUN mkdir -p /app/chroma_db

ENV PORT=8002

# Gunicorn как прод-сервер
CMD ["gunicorn", "-b", "0.0.0.0:8002", "app:app"]