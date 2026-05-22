FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# 1. Al usar python:3.12-slim, los repositorios están sanos y el apt-get update funcionará sin error 100.
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 tzdata build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalamos PyTorch con soporte para CUDA exactamente como indica el README de Meta
RUN pip install --no-cache-dir torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Resto de dependencias de tu API
RUN pip install --no-cache-dir runpod opencv-python-headless numpy transformers huggingface_hub pillow setuptools wheel

# 4. Instalamos SAM 3 desde el repositorio oficial
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e .

# 5. Configuramos el Token de Hugging Face y descargamos los pesos de la IA
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN} 
RUN python -c "from huggingface_hub import login; login(token='$HF_TOKEN'); from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"

COPY app.py .

CMD ["python", "-u", "app.py"]
