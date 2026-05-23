import runpod
import os
import torch
import numpy as np
import base64
import cv2
import requests

from huggingface_hub import login
from transformers import AutoModel, AutoProcessor

# -------------------------------------------------
# CONFIG ESTABLE
# -------------------------------------------------
torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE, flush=True)

# -------------------------------------------------
# HF LOGIN
# -------------------------------------------------
print("Logging into HuggingFace...", flush=True)
login(token=os.environ.get("HF_TOKEN", ""))
print("HF OK", flush=True)

# -------------------------------------------------
# LOAD SAM3 (runtime only)
# -------------------------------------------------
print("Loading SAM3...", flush=True)

processor = AutoProcessor.from_pretrained(
    "facebook/sam3",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "facebook/sam3",
    trust_remote_code=True,
    torch_dtype=torch.float32
).to(DEVICE)

model.eval()

print("SAM3 loaded", flush=True)

# -------------------------------------------------
# IMAGE LOADER
# -------------------------------------------------
def load_image(data):
    if data.startswith("http"):
        r = requests.get(data, timeout=20)
        arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        if data.startswith("data:image"):
            data = data.split(",")[1]
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(data), np.uint8),
            cv2.IMREAD_COLOR
        )

    if img is None:
        raise ValueError("Invalid image")

    # CORRECCIÓN 1: Convertir de BGR a RGB para que Hugging Face entienda los colores
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------------------------------------------------
# RESIZE SAFE
# -------------------------------------------------
def resize(img, max_size=1024):
    h, w = img.shape[:2]
    scale = min(max_size / w, max_size / h)

    if scale >= 1:
        return img

    return cv2.resize(img, (int(w * scale), int(h * scale)))

# -------------------------------------------------
# POST-PROCESSING PARA ANDROID
# -------------------------------------------------
def get_bbox_from_mask(mask_2d):
    """Calcula el Bounding Box [x_min, y_min, x_max, y_max] buscando los bordes de la máscara"""
    y_indices, x_indices = np.where(mask_2d > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0] # Si no detectó nada, devuelve 0
    
    return [
        int(np.min(x_indices)), 
        int(np.min(y_indices)), 
        int(np.max(x_indices)), 
        int(np.max(y_indices))
    ]

def encode_mask_to_base64(mask_2d):
    """Convierte la matriz a un PNG con fondo transparente y máscara blanca"""
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

        img = resize(load_image(inp["file"]), 1024)

        text = inp.get("text", "object")

        pattern = inp.get("pattern", None)

        print("Text:", text, flush=True)

        # -------------------------------------------------
        # TEXT + IMAGE PROCESSING (A PRUEBA DE FALLOS)
        # -------------------------------------------------
        # CORRECCIÓN 2: Bloque try/except para evadir el error de Meta con "text" vs "prompt"
        try:
            inputs = processor(
                images=img,
                text=text,
                return_tensors="pt"
            )
        except TypeError:
            inputs = processor(
                images=img,
                prompt=text,
                return_tensors="pt"
            )

        # -------------------------------------------------
        # MOVER A GPU (SAFE CASTING)
        # -------------------------------------------------
        # CORRECCIÓN 3: Respetamos los enteros para el texto y pasamos a float32 solo la imagen
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].float()

        # CORRECCIÓN 4: Indentación correcta
        with torch.no_grad():
            outputs = model(**inputs)

        # -------------------------------------------------
        # EXTRACCIÓN DE RESULTADOS
        # -------------------------------------------------
        # 1. Localizar los tensores de las máscaras (Hugging Face puede llamarlas distinto)
        raw_masks = getattr(outputs, "pred_masks", None)
        if raw_masks is None:
            raw_masks = outputs.get("pred_masks", outputs.get("masks"))
            
        if raw_masks is None:
            raise ValueError(f"No se encontraron máscaras en el output. Keys disponibles: {list(outputs.keys())}")

        # 2. Pasamos a CPU y a Numpy
        masks_np = raw_masks.detach().cpu().numpy()

        # 3. SAM suele devolver arrays de 4 o 5 dimensiones [Batch, Prompts, Niveles, H, W]
        # Vamos a ir quitando dimensiones hasta quedarnos con la matriz 2D de la máscara
        while masks_np.ndim > 2:
            masks_np = masks_np[0]
            
        mask_2d = masks_np
        
        # Si la máscara viene como logits (números decimales), aplicamos umbral > 0
        if mask_2d.dtype != bool:
            mask_2d = mask_2d > 0.0

        # 4. Redimensionamos la máscara de vuelta al tamaño de la imagen que metimos
        orig_h, orig_w = img.shape[:2]
        # Usamos INTER_NEAREST para no crear grises borrosos en los bordes de la máscara
        mask_resized = cv2.resize(mask_2d.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 5. Generamos la caja y la imagen Base64 para Android
        bbox = get_bbox_from_mask(mask_resized)
        mask_b64 = encode_mask_to_base64(mask_resized)

        # Intentamos extraer el score de confianza si existe, si no, devolvemos 1.0
        score = 1.0
        raw_scores = getattr(outputs, "iou_scores", outputs.get("iou_scores"))
        if raw_scores is not None:
            score = float(raw_scores.detach().cpu().numpy().flatten()[0])

        return {
            "status": "success",
            "prompt": text,
            "detections": [{
                "box": bbox,
                "score": round(score, 3),
                "mask_base64": mask_b64
            }]
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
