# Image LEGERE : correspond au requirements.txt "minimal" (pas de RAG/
# ChromaDB/sentence-transformers, qui tirent torch et ~1-2 Go) -- le backend
# degrade deja proprement sans eux (RAG_AVAILABLE=False, voir agent.py).
# Build context = racine du depot (pas mae_backend/), pour pouvoir copier
# processed_data/ et outputs/ en plus du code : main.py les lit via des
# chemins relatifs ("../processed_data/...", "../outputs/...") suppose
# executes depuis mae_backend/, d'ou le WORKDIR ci-dessous.
#
#   docker build -f mae_backend/Dockerfile -t mae-backend .
#   docker run -p 8000:8000 --env-file mae_backend/.env mae-backend
FROM python:3.11-slim

WORKDIR /app

COPY mae_backend/requirements.txt mae_backend/requirements.txt
RUN pip install --no-cache-dir -r mae_backend/requirements.txt

COPY mae_backend/ mae_backend/
COPY processed_data/ processed_data/
COPY outputs/ outputs/

WORKDIR /app/mae_backend

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
