FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY app_factory.py /app/
COPY config.py /app/
COPY domain/ /app/domain/
COPY jobs/ /app/jobs/
COPY services/ /app/services/
COPY _golden /app/_golden

# Папка для базы Chroma
# В docker-compose смонтируем сюда volume
RUN mkdir -p /app/chroma_db

# Логи в Docker лучше в stdout -> logging StreamHandler по умолчанию подходит
CMD ["python", "-m", "app"]
