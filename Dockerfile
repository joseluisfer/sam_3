# Usamos una imagen base oficial de PyTorch con soporte para CUDA (GPU)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 1. Dependencias del sistema (críticas para OpenCV)
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalamos las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Descargamos los pesos de SAM 3 usando tu token de Hugging Face
# Declaramos el ARG. Esto permite pasar el token en el momento del build sin guardarlo en la imagen.
ARG HF_TOKEN
RUN mkdir -p /app/weights && \
    huggingface-cli download meta-llama/sam3 sam3.safetensors --local-dir /app/weights --token $HF_TOKEN

# 4. Copiamos tu script de RunPod
COPY app.py .

# 5. Arrancamos el worker
CMD ["python", "-u", "app.py"]
