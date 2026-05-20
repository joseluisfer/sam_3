FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Declarar que recibiremos un token durante la construcción
ARG HF_TOKEN
# 2. Convertirlo en variable de entorno para que transformers lo use
ENV HF_TOKEN=$HF_TOKEN

# 3. Descargar el modelo (detectará el ENV HF_TOKEN automáticamente)
RUN python -c "from transformers import Sam3Model, Sam3Processor; Sam3Model.from_pretrained('facebook/sam3'); Sam3Processor.from_pretrained('facebook/sam3')"

COPY handler.py .

CMD ["python", "-u", "/handler.py"]
