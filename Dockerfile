# Usamos Python puro para evitar los repositorios rotos
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

# 2. Instalamos tus librerías de requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instalamos SAM 3 nativo desde el GitHub oficial
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir .

# 4. Copiamos tu código
COPY handler.py .

CMD ["python", "-u", "handler.py"]
