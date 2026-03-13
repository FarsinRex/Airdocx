#base image
#CHOOSING SLIM-BOOKWORM AS IT IS DEBIAN BASED - SENTENCE TRANSFORMERS HAS C DEPENDENCIES
FROM python:3.11-slim-bookworm

#system dependencies
#As pdfplumber needs libgomp(OpenMP) for some rendering ops. installing it via apt
#combining it into one run to reduce cacheing and bloating of the image
RUN apt-get update && apt-get install -y \
libgomp1 \
&& rm -rf /var/lib/apt/lists/*

#working directory
WORKDIR /app

#installing python dependencies
#copying req.txt first before rest of the code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#download the model during the build, so that it will be baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
#copying application code
#.env will be injected at runtime via docker-compose env_file
COPY . .

#uploads directory
RUN mkdir -p /uploads

#port
EXPOSE 8000
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
#startup command
CMD ["uvicorn","main:app", "--host", "0.0.0.0", "--port", "8000"]


