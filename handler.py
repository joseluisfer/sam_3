import runpod
import numpy as np
import base64
import cv2
import requests
import sys
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

print("Iniciando contenedor y cargando SAM3...", flush=True)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model.eval()
    print(f"Modelo SAM3 cargado en {device}", flush=True)
except Exception as e:
    print(f"ERROR CRÍTICO AL CARGAR SAM3: {str(e)}", flush=True)
    sys.exit(1)

# -------------------------
# Loader de Imágenes (idéntico al tuyo)
# -------------------------
def load_image(data):
    if not isinstance(data, str):
        raise ValueError("Input must be string")
    
    if data.startswith("http"):
        print(f"Descargando URL: {data[:60]}...", flush=True)
        resp = requests.get(data, timeout=10)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        print("Decodificando Base64...", flush=True)
        if data.startswith("data:image"):
            data = data.split(",")[1]
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Decodificación fallida")
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # SAM3 espera RGB

# -------------------------
# SAM3 específico: preparar prompts
# -------------------------
def prepare_prompts(points=None, boxes=None):
    """Convierte prompts estilo YOLOE a lo que espera SAM3"""
    prompts = {}
    if points is not None:
        prompts["point_coords"] = torch.tensor(points, dtype=torch.float)
        prompts["point_labels"] = torch.ones(len(points), dtype=torch.long)
    if boxes is not None:
        prompts["boxes"] = torch.tensor(boxes, dtype=torch.float)
    return prompts

# -------------------------
# Handler principal
# -------------------------
def handler(job):
    try:
        input_data = job["input"]
        print(f"--- TRABAJO {job['id']} ---", flush=True)
        
        # 1. Imagen obligatoria (igual que en YOLOE)
        if "file" not in input_data:
            return {"error": "Falta 'file' (imagen)"}
        
        img = load_image(input_data["file"])
        
        # 2. SAM3 usa puntos o cajas (no texto nativo como YOLOE)
        # Aquí la diferencia clave: SAM3 necesita coordenadas
        points = input_data.get("points")  # ej: [[100, 150], [200, 250]]
        boxes = input_data.get("boxes")    # ej: [[50, 50, 200, 300]]
        
        if not points and not boxes:
            return {"error": "SAM3 necesita 'points' o 'boxes' como prompt"}
        
        # 3. Procesar con SAM3
        prompts = prepare_prompts(points, boxes)
        inputs = processor(images=img, **prompts, return_tensors="pt")
        
        # Mover a GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 4. Extraer máscaras (similar a las detecciones de YOLOE)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Convertir a formato similar a tus detecciones
        detections = []
        for i, mask in enumerate(masks[0]):
            # Calcular bounding box de la máscara
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                bbox = [
                    int(x_coords.min()), int(y_coords.min()),
                    int(x_coords.max()), int(y_coords.max())
                ]
                detections.append({
                    "bbox": bbox,
                    "confidence": float(outputs.iou_scores[0][i].cpu()) if hasattr(outputs, 'iou_scores') else 1.0,
                    "mask_size": int(mask.sum())
                })
        
        print(f"Máscaras generadas: {len(detections)}", flush=True)
        
        return {
            "status": "success",
            "count": len(detections),
            "detections": detections
        }
    
    except Exception as e:
        print(f"ERROR: {str(e)}", flush=True)
        return {"error": str(e)}

# -------------------------
# Arranque serverless
# -------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
