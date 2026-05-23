# Usamos Python puro para evitar los repositorios rotos (Error 100)
FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app/cache

# Instalamos dependencias de sistema necesarias para OpenCV y compilar
RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

# 1. Instalamos el PyTorch 2.10.0 exacto que pide SAM 3
RUN pip install --no-cache-dir torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Instalamos tus librerías de requirements.txt (runpod, opencv, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instalamos SAM 3 nativo desde el GitHub oficial
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir .

# 4. PRE-DESCARGA DEL MODELO (El secreto para que RunPod sea rápido)
# Necesitas pasar el HF_TOKEN en los secretos de GitHub Actions
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from huggingface_hub import login; login(token='$HF_TOKEN'); from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
