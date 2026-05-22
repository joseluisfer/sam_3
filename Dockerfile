FROM python:3.12
ENV TZ=Europe/Madrid
WORKDIR /app

# 1. PyTorch con CUDA
RUN pip install --no-cache-dir torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Dependencias base — numpy 1.26 primero porque sam3 exige numpy<2
#    opencv se instala DESPUÉS para que no lo sobreescriba
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    pillow \
    runpod \
    transformers \
    huggingface_hub \
    setuptools wheel

# 3. SAM 3 desde el repo oficial (necesita numpy 1.26 instalado antes)
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    pip install --no-cache-dir -e .

# 4. OpenCV DESPUÉS de sam3 para que no rompa numpy
#    --no-deps evita que reinstale numpy>=2
RUN pip install --no-cache-dir --no-deps opencv-python-headless

# 5. Copiamos el handler
# ⚠️  NO pre-descargamos el modelo aquí: el HF_TOKEN solo existe en runtime,
#     no durante docker build. La descarga ocurre en app.py al arrancar.
COPY app.py .

CMD ["python", "-u", "app.py"]
