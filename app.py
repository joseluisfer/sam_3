import runpod
import base64
import numpy as np
import cv2
import torch
import io
from PIL import Image

# Importamos las clases exactas que dicta el README oficial
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print("Iniciando contenedor y cargando SAM 3...", flush=True)

try:
    # Según el README, esta función inicializa el modelo.
    # Al estar logueados con Hugging Face en el contenedor, esto cargará los pesos automáticamente.
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("¡Modelo SAM 3 cargado con éxito!", flush=True)
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}", flush=True)


def encode_mask_to_base64(mask_array):
    """Convierte la máscara (matriz booleana) a un PNG transparente para Android"""
    # SAM 3 devuelve tensores en la GPU. Los pasamos a la CPU y a Numpy
    mask_np = mask_array.cpu().numpy()
    
    # Manejo de dimensiones extra (a veces devuelve [1, H, W])
    if len(mask_np.shape) == 3:
        mask_np = mask_np[0]
        
    h, w = mask_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Pintamos la máscara de blanco semitransparente
    rgba[mask_np > 0] = [255, 255, 255, 150] 
    
    _, buffer = cv2.imencode('.png', rgba)
    return base64.b64encode(buffer).decode('utf-8')


def handler(job):
    job_input = job.get('input', {})
    image_b64 = job_input.get('image_base64')
    text_prompt = job_input.get('text_prompt')
    
    if not image_b64 or not text_prompt:
        return {"error": "Faltan campos obligatorios: 'image_base64' o 'text_prompt'"}
        
    try:
        # 1. Limpiar cabecera Base64 si existe
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
            
        # 2. Convertir Base64 a imagen PIL (El formato que exige SAM 3)
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 3. Inferencia según el manual oficial
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        # 4. Extraer resultados
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        resultados = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                resultados.append({
                    "box": boxes[i].cpu().numpy().tolist(), 
                    "score": float(scores[i].cpu().numpy()),
                    "mask_base64": encode_mask_to_base64(masks[i])
                })
            
        return {
            "status": "success",
            "prompt": text_prompt,
            "detections": resultados
        }
        
    except Exception as e:
        return {"error": f"Fallo en la inferencia: {str(e)}"}

runpod.serverless.start({"handler": handler})
