FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Код приложения
COPY src/ ./src/
COPY config.yaml ./config.yaml

# 3) Служебные директории
RUN mkdir -p models data/processed

# Чтобы импорты `src.*` работали
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1


EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host ", "0.0.0.0", "--port", "8000"]