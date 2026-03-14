FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/
COPY app_factory.py /app/
COPY config.py /app/
COPY clients/ /app/clients/
COPY eval/ /app/eval/
COPY jobs/ /app/jobs/
COPY models/ /app/models/
COPY storage/ /app/storage/

# Логи в Docker лучше в stdout -> logging StreamHandler по умолчанию подходит
CMD ["python", "-m", "app"]
