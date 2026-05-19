FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 🔥 Forzar a Hugging Face a usar la carpeta de caché controlada
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# 🛠️ CREACIÓN PREVIA: Asegura que la carpeta exista antes de que Python escriba en ella
RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# 🔥 Precarga absoluta en una sola línea (SAM 3 pesa ~3.4GB)
# ARG permite pasar el token durante el build para descargar los pesos oficiales
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN
RUN python -c "import os; from huggingface_hub import login; login(os.environ.get('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')) if os.environ.get('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') else None; from transformers import Sam3Processor, Sam3Model; Sam3Processor.from_pretrained('facebook/sam3'); Sam3Model.from_pretrained('facebook/sam3')"

# 🛠️ PERMISOS TOTALES: Permite que el usuario dinámico de RunPod lea la caché
RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
