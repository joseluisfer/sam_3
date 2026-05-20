FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

# Variables de entorno para cachear modelos
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache

# Crear carpeta de caché ANTES de precargar
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 🔥 PRECARGA COMPLETA de SAM3 (TODO en UNA sola línea)
RUN python -c "import torch; import numpy as np; from transformers import SamModel, SamProcessor; print('Precargando SAM3...'); model = SamModel.from_pretrained('facebook/sam-vit-huge'); processor = SamProcessor.from_pretrained('facebook/sam-vit-huge'); model.eval(); dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8); inputs = processor(images=dummy_image, return_tensors='pt'); with torch.no_grad(): _ = model(**inputs); print('Precarga completada')"

# Permisos para el usuario dinámico de RunPod
RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
