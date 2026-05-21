FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 1. Instalar dependencias del sistema operativo (Git y wget para los pesos)
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# 2. Configurar variables de entorno para las cachés
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache
RUN mkdir -p /app/cache /app/weights

# 3. Clonar e instalar el repositorio oficial de Segment Anything de Meta
RUN git clone https://github.com/facebookresearch/sam2.git /app/sam2_repo
WORKDIR /app/sam2_repo
RUN pip install -e .
WORKDIR /app

# 4. Descargar los pesos oficiales optimizados (SAM 2.1 Large / SAM 3 Preview)
# Usamos el checkpoint 'sam2.1_hiera_large.pt' que es el motor de la demo actual
RUN wget -O /app/weights/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/012124/sam2.1_hiera_large.pt

# 5. Copiar los archivos de tu proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# Dar permisos totales a las carpetas de trabajo
RUN chmod -R 777 /app/cache /app/weights

CMD ["python", "-u", "handler.py"]
