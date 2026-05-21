FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 🔥 Forzar carpetas de caché públicas
ENV YOLO_CONFIG_DIR=/app/cache
ENV RUNNY_CACHE_DIR=/app/cache
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el script de descarga y el handler
COPY download_models.py .
COPY handler.py .

# 🔥 Ejecutamos el script de precarga de forma limpia
RUN python download_models.py

RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
