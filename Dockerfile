FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 tzdata build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Dependencias de Python. Instalamos la versión específica de PyTorch que pide el README
RUN pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip install runpod opencv-python-headless numpy transformers huggingface_hub pillow setuptools wheel

# 2. Instalamos SAM 3 desde el repositorio oficial
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e .

# 3. Configuramos el Token de Hugging Face de forma segura
ARG HF_TOKEN
# Convertimos el argumento de build en una variable de entorno para que Python la vea
ENV HF_TOKEN=${HF_TOKEN} 

# 4. Magia negra de Docker: Ejecutamos Python un segundo para obligar a SAM 3 a 
# descargar los pesos pesados ahora, en lugar de hacerlo durante el Cold Start de RunPod.
RUN python -c "from huggingface_hub import login; login(token='$HF_TOKEN'); from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"

COPY app.py .

CMD ["python", "-u", "app.py"]
