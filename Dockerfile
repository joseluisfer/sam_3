FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

RUN mkdir -p /app/cache

RUN apt-get update && apt-get install -y \
    git wget libgl1 libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
