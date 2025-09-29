FROM python:3.11-slim

WORKDIR /app
COPY ./src /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn pypdf sentence-transformers faiss-cpu

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
