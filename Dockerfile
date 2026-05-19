# Usamos la imagen oficial de RunPod con PyTorch y CUDA
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 🔥 Forzar a Hugging Face y PyTorch a usar la carpeta de caché controlada
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# 🛠️ CREACIÓN PREVIA: Asegura que la carpeta exista antes de que Python escriba en ella
RUN mkdir -p /app/cache

# Copiamos e instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos nuestro script del servidor
COPY handler.py .

# 🔥 PRECARGA DEL MODELO SAM 3
# Declaramos la variable. RunPod inyectará el valor desde su interfaz web.
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Descarga el modelo usando la variable de entorno
RUN python -c "import os; from huggingface_hub import login; login(os.environ.get('HF_TOKEN')) if os.environ.get('HF_TOKEN') else None; from transformers import Sam3Processor, Sam3Model; Sam3Processor.from_pretrained('facebook/sam3'); Sam3Model.from_pretrained('facebook/sam3')"

# 🛠️ PERMISOS TOTALES: Permite que el usuario dinámico de RunPod lea la caché sin descargar nada de nuevo
RUN chmod -R 777 /app/cache

# Comando de inicio de RunPod Serverless
CMD ["python", "-u", "handler.py"]
