FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy Python dependencies
COPY requirements.txt .

# Install CPU-only PyTorch to avoid CUDA/cuDNN packages
RUN pip install --no-cache-dir \
    torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src ./src
COPY app.py .

# Copy processed RAG data
COPY data/processed/zx_bank_documents.jsonl ./data/processed/
COPY data/processed/zx_bank_embeddings.json ./data/processed/

# Copy FAISS index and metadata
COPY data/indexes ./data/indexes

# FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]