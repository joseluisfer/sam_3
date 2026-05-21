import runpod
import base64
import cv2
import numpy as np
import torch
from sam3 import SAM3Registry, SAM3Predictor

print("Iniciando contenedor y cargando SAM 3 en VRAM...", flush=True)

# 1. Cargar el modelo a nivel global (Evita el Cold Start en cada petición)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Ajusta la ruta y el modelo según la nomenclatura exacta del repo de SAM 3
sam3_model = SAM3Registry.build_model(checkpoint="/app/weights/sam3.safetensors")
sam3_model.to(device=device)
predictor = SAM3Predictor(sam3_model)

print("¡Modelo cargado con éxito!", flush=True)

def decode_base64(b64_string):
    """Convierte Base64 a imagen OpenCV"""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_mask_to_base64(mask_array):
    """Convierte la máscara generada a un PNG transparente en Base64 para Android"""
    # Crear una imagen RGBA donde la máscara es blanca y el resto transparente
    h, w = mask_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_array > 0] = [255, 255, 255, 150] # Blanco con opacidad
    
    _, buffer = cv2.imencode('.png', rgba)
    return base64.b64encode(buffer).decode('utf-8')

def handler(job):
    job_input = job.get('input', {})
    image_b64 = job_input.get('image_base64')
    text_prompt = job_input.get('text_prompt')
    
    if not image_b64 or not text_prompt:
        return {"error": "Faltan campos: image_base64 o text_prompt"}
        
    try:
        # Preparar la imagen
        image_bgr = decode_base64(image_b64)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Inferencia con SAM 3 usando texto
        predictor.set_image(image_rgb)
        masks, boxes, scores = predictor.predict_with_text(text_prompt=text_prompt)
        
        # Formatear la respuesta para Android
        resultados = []
        for i in range(len(boxes)):
            resultados.append({
                "box": boxes[i].tolist(), # [x_min, y_min, x_max, y_max]
                "score": float(scores[i]),
                "mask_base64": encode_mask_to_base64(masks[i])
            })
            
        return {
            "status": "success",
            "prompt": text_prompt,
            "detections": resultados
        }
        
    except Exception as e:
        return {"error": str(e)}

# Iniciar RunPod
runpod.serverless.start({"handler": handler})
