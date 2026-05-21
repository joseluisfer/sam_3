import runpod
import os
import torch
import numpy as np
import cv2
import base64
import requests
from PIL import Image
import sys

# Añadir el repositorio clonado al PATH de Python por seguridad
sys.path.append("/app/sam2_repo")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Dispositivo detectado por RunPod: {device}", flush=True)

# Inicializar el predictor oficial de Meta con los pesos descargados
try:
    print("⏳ Cargando arquitectura SAM oficial en VRAM...", flush=True)
    # El archivo de configuración se encuentra dentro del repo clonado
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path = "/app/weights/sam2.1_hiera_large.pt"
    
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ ¡Modelo cargado y listo en memoria!", flush=True)
except Exception as e:
    print(f"❌ Error crítico cargando el modelo oficial: {str(e)}", flush=True)
    sys.exit(1)

def load_image(data):
    """Descarga o decodifica la imagen de entrada."""
    if data.startswith("http"):
        resp = requests.get(data, timeout=15)
        resp.raise_for_status()
        img_bytes = resp.content
    else:
        if "base64," in data:
            data = data.split("base64,")[1]
        img_bytes = base64.b64decode(data)
        
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def mask_to_base64(mask_array):
    """Convierte la máscara booleana a PNG codificado en Base64."""
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')

def handler(job):
    try:
        input_data = job["input"]
        if "file" not in input_data:
            return {"error": "Falta el campo requerido 'file'"}

        # 1. Cargar la imagen y pasarla al predictor de Meta
        image_rgb = load_image(input_data["file"])
        predictor.set_image(image_rgb)

        # 2. Leer los tipos de interacciones (Prompts de clics o recuadros)
        points = input_data.get("points")   # Formato: [[x1, y1], [x2, y2]]
        labels = input_data.get("labels")   # Formato: [1, 0] (1=incluir, 0=excluir)
        box = input_data.get("box")         # Formato: [x_min, y_min, x_max, y_max]

        input_points = np.array(points) if points else None
        input_labels = np.array(labels) if labels else None
        input_box = np.array(box) if box else None

        if input_points is None and input_box is None:
            return {"error": "Debes enviar coordenadas en 'points' o un recuadro en 'box' para segmentar."}

        # 3. Ejecutar la inferencia de la arquitectura de Meta
        print("🔮 Ejecutando segmentación...", flush=True)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=False # Devolver la mejor máscara consolidada
        )

        # 4. Formatear la salida para la API
        # masks tiene forma [N, H, W], extraemos la máscara principal
        best_mask = masks[0]
        best_score = float(scores[0])

        return {
            "status": "success",
            "confidence": best_score,
            "mask_base64": mask_to_base64(best_mask)
        }

    except Exception as e:
        print(f"❌ Error en el Job Handler: {str(e)}", flush=True)
        return {"error": str(e)}

# Iniciar el bucle de RunPod
runpod.serverless.start({"handler": handler})
