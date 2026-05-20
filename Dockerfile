FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

RUN chmod -R 777 /app/cache

CMD ["python", "-u", "handler.py"]
