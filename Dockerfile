FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 🔥 Forzar a Hugging Face y PyTorch a usar la misma carpeta pública
ENV YOLO_CONFIG_DIR=/app/cache
ENV RUNNY_CACHE_DIR=/app/cache
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# 🛠️ CREACIÓN PREVIA: Asegura que la carpeta exista antes de que Python escriba en ella
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 🔥 PRECARGA DINÁMICA: Descarga SAM 3 usando el token directamente desde Python
RUN python -c " \
import os, sys; \
try: \
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor; \
    repo = 'facebook/sam3'; \
    \
    print('-> Iniciando precarga de modelos SAM 3...'); \
    Sam3Model.from_pretrained(repo); \
    Sam3Processor.from_pretrained(repo); \
    Sam3TrackerModel.from_pretrained(repo); \
    Sam3TrackerProcessor.from_pretrained(repo); \
    print('✅ Todo se ha precargado correctamente en /app/cache'); \
except Exception as e: \
    print(f'❌ ERROR CRÍTICO EN PRECARGA: {str(e)}', file=sys.stderr); \
    sys.exit(1); \
"

# 🛠️ PERMISOS TOTALES: Permite que el usuario dinámico de RunPod lea la caché sin descargar nada
RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
