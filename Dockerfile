FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid

# Añadimos 'build-essential' que incluye los compiladores de C++ necesarios para compilar SAM 3
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 tzdata build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Paso 1: Instalar dependencias base de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Paso 2: Clonar e instalar SAM 3 de forma aislada para evitar el error de compilación
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e .

# Paso 3: Descargar los pesos usando tu token de Hugging Face
ARG HF_TOKEN
RUN mkdir -p /app/weights && \
    huggingface-cli download meta-llama/sam3 sam3.safetensors --local-dir /app/weights --token $HF_TOKEN

COPY app.py .

CMD ["python", "-u", "app.py"]
