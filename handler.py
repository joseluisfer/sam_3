import runpod
import os
import torch
import numpy as np
import base64
import cv2
import requests
import io
from PIL import Image

from huggingface_hub import login
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# -------------------------------------------------
# HF LOGIN (Obligatorio para descargar los pesos)
# -------------------------------------------------
print("Logging into HuggingFace...", flush=True)
login(token=os.environ.get("HF_TOKEN", ""))
print("HF OK", flush=True)

# -------------------------------------------------
# LOAD SAM3 (Nativo de Meta)
# -------------------------------------------------
print("Loading SAM3 Native API...", flush=True)

# Usamos la API oficial de Meta, no la de Hugging Face Transformers
model = build_sam3_image_model()
processor = Sam3Processor(model)

print("SAM3 loaded successfully", flush=True)

# -------------------------------------------------
# IMAGE LOADER (Ahora usamos PIL como exige Meta)
# -------------------------------------------------
def load_image_pil(data):
    if data.startswith("http"):
        r = requests.get(data, timeout=20)
        image_bytes = r.content
    else:
        if data.startswith("data:image"):
            data = data.split(",")[1]
        image_bytes = base64.b64decode(data)

    # Meta exige formato PIL y en RGB nativo
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize por seguridad (Max 1024px)
    max_size = 1024
    w, h = image.size
    scale = min(max_size / w, max_size / h)
    
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image

# -------------------------------------------------
# POST-PROCESSING PARA ANDROID
# -------------------------------------------------
def encode_mask_to_base64(mask_2d):
    """Convierte la matriz numpy a un PNG con fondo transparente y máscara blanca"""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Rellenamos la máscara con color blanco (255, 255, 255) y 60% opacidad (150)
    rgba[mask_2d > 0] = [255, 255, 255, 150] 
    
    _, buffer = cv2.imencode('.png', rgba)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------------------------------
# HANDLER
# -------------------------------------------------
def handler(job):
    try:
        inp = job["input"]
        
        # 1. Cargamos el texto y la imagen
        text_prompt = inp.get("text", "object")
        print(f"Buscando: {text_prompt}", flush=True)
        
        image = load_image_pil(inp["file"])

        # -------------------------------------------------
        # INFERENCIA NATIVA (Exactamente como dicta el README)
        # -------------------------------------------------
        # Cargamos la imagen en el cerebro del procesador
        inference_state = processor.set_image(image)
        
        # Le enviamos el texto (usando la variable nativa 'prompt')
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        # -------------------------------------------------
        # EXTRACCIÓN DE RESULTADOS
        # -------------------------------------------------
        # La API nativa ya nos da exactamente lo que queremos bien ordenado
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        resultados = []
        
        if boxes is not None and len(boxes) > 0:
            # Iteramos sobre todas las detecciones que haya encontrado
            for i in range(len(boxes)):
                # Extraemos la máscara matemática a la CPU y limpiamos dimensiones
                mask_np = masks[i].cpu().numpy()
                while mask_np.ndim > 2:
                    mask_np = mask_np[0]
                    
                # Convertimos a Base64
                mask_b64 = encode_mask_to_base64(mask_np)
                
                # Extraemos coordenadas de la caja y confianza
                box_np = boxes[i].cpu().numpy().tolist()
                score_val = float(scores[i].cpu().numpy())
                
                resultados.append({
                    "box": box_np,
                    "score": round(score_val, 3),
                    "mask_base64": mask_b64
                })

        return {
            "status": "success",
            "prompt": text_prompt,
            "detections": resultados
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# -------------------------------------------------
# RUNPOD
# -------------------------------------------------
runpod.serverless.start({"handler": handler})
