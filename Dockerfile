FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 🔥 Forzar a Hugging Face y PyTorch a usar la misma carpeta pública
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
ENV RUNNY_CACHE_DIR=/app/cache

# 🛠️ CREACIÓN PREVIA: Asegura que la carpeta exista antes de escribir en ella
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 🔐 ARGUMENTO DE CONSTRUCCIÓN: Por si el repositorio requiere autenticación
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 🔥 Precarga absoluta en una sola línea de los modelos de SAM 3
RUN python -c "import os; from huggingface_hub import login; token = os.environ.get('HF_TOKEN'); \
if token: login(token=token); \
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor; \
repo = 'facebook/sam3'; \
Sam3Model.from_pretrained(repo); Sam3Processor.from_pretrained(repo); \
Sam3TrackerModel.from_pretrained(repo); Sam3TrackerProcessor.from_pretrained(repo)"

# 🛠️ PERMISOS TOTALES
RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
