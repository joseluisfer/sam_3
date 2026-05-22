# -------------------------------------------------
# Base PyTorch + CUDA
# -------------------------------------------------
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# -------------------------------------------------
# Cache HF/Torch
# -------------------------------------------------
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HUGGINGFACE_HUB_CACHE=/app/cache

RUN mkdir -p /app/cache

# -------------------------------------------------
# Dependencias sistema
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libgl1

# -------------------------------------------------
# Python deps
# -------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Handler
# -------------------------------------------------
COPY handler.py .

# -------------------------------------------------
# Precarga SAM3
# HF_TOKEN debe existir en build args
# -------------------------------------------------
ARG HF_TOKEN

RUN python -c " \
from huggingface_hub import login; \
from transformers import AutoModel; \
import torch; \
login(token='${HF_TOKEN}'); \
print('Downloading SAM3...'); \
model = AutoModel.from_pretrained( \
    'facebook/sam3', \
    trust_remote_code=True, \
    torch_dtype=torch.float32 \
); \
print('SAM3 downloaded successfully'); \
"

# -------------------------------------------------
# Permisos cache
# -------------------------------------------------
RUN chmod -R 777 /app/cache

# -------------------------------------------------
# Launch
# -------------------------------------------------
CMD ["python", "-u", "handler.py"]
