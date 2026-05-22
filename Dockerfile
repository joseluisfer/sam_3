# Usamos la imagen completa que ya trae Git y compiladores C++ de fábrica
FROM python:3.12

ENV TZ=Europe/Madrid

WORKDIR /app

# 1. Instalamos PyTorch con soporte para CUDA (GPU)
RUN pip install --no-cache-dir torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Resto de dependencias de tu API (usamos opencv headless para no depender de librerías del sistema)
RUN pip install --no-cache-dir runpod opencv-python-headless numpy transformers huggingface_hub pillow setuptools wheel

# 3. Instalamos SAM 3 desde el repositorio oficial
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e .

# 4. Configuramos el Token de Hugging Face y descargamos el "cerebro"
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN} 
RUN python -c "from huggingface_hub import login; login(token='$HF_TOKEN'); from sam3.model_builder import build_sam3_image_model; build_sam3_image_model()"

# 5. Copiamos tu código
COPY app.py .

CMD ["python", "-u", "app.py"]
