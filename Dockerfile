FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

RUN mkdir -p /app/cache

# 🔥 ACTUALIZAMOS A PYTORCH 2.5.1 PARA SOLUCIONAR EL ERROR DE INFER_SCHEMA 🔥
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Instalamos el resto de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
