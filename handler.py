import runpod
import numpy as np
import base64
import cv2
import requests
import sys
import os
import torch
from PIL import Image
from transformers import (
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor
)
from huggingface_hub import login

# -------------------------
# Inicialización y Carga de Modelos
# -------------------------
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("Autenticando con Hugging Face Hub...", flush=True)
    login(token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Usando dispositivo de cómputo: {device}", flush=True)

MODEL_REPO = "facebook/sam3"

try:
    print("⏳ Cargando modelos SAM3 permanentemente en VRAM...", flush=True)
    # 1. Modelo de Segmentación por Texto
    IMG_MODEL = Sam3Model.from_pretrained(MODEL_REPO).to(device)
    IMG_PROCESSOR = Sam3Processor.from_pretrained(MODEL_REPO)
    
    # 2. Modelo de Segmentación por Clics (Tracker)
    TRK_MODEL = Sam3TrackerModel.from_pretrained(MODEL_REPO).to(device)
    TRK_PROCESSOR = Sam3TrackerProcessor.from_pretrained(MODEL_REPO)
    
    print("✅ ¡Todos los modelos SAM3 cargados exitosamente!", flush=True)
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR LOS MODELOS: {str(e)}", flush=True)
    sys.exit(1)

# -------------------------
# Loader de Imágenes
# -------------------------
def load_image(data):
    if not isinstance(data, str):
        raise ValueError("El input de la imagen debe ser un string (URL o Base64)")

    if data.startswith("http"):
        print(f"Descargando URL: {data[:60]}...", flush=True)
        try:
            resp = requests.get(data, timeout=10) 
            resp.raise_for_status()
            img_array = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise RuntimeError(f"Error de red descargando imagen: {str(e)}")
    else:
        print("Decodificando cadena Base64...", flush=True)
        if data.startswith("data:image"):
            data = data.split(",")[1]
        img_bytes = base64.b64decode(data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error al decodificar la imagen (cv2 devolvió None)")

    # SAM 3 trabaja con imágenes PIL en formato RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def mask_to_base64(mask_array):
    """Convierte una máscara booleana numpy a un string Base64 PNG."""
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')

# -------------------------
# Handler del Ciclo de Vida del Job
# -------------------------
def handler(job):
    try:
        input_data = job["input"]
        print(f"--- NUEVO TRABAJO RECIBIDO (ID: {job['id']}) ---", flush=True)

        if "file" not in input_data:
            return {"error": "Falta el campo obligatorio 'file' (imagen principal)"}

        text_query = input_data.get("text")
        points_query = input_data.get("points") # Espera formato: [[x1, y1], [x2, y2]]

        if not text_query and not points_query:
            return {"error": "Debes proporcionar al menos un prompt de 'text' o coordenadas en 'points'"}

        # Cargar imagen de entrada
        pil_image = load_image(input_data["file"])
        
        # Estructura final de respuesta
        output_masks = []

        # ==========================================
        # CASO A: Segmentación por TEXTO (Open-Vocab)
        # ==========================================
        if text_query:
            print(f"Modo: Segmentación por texto -> Prompt: '{text_query}'", flush=True)
            conf_thresh = input_data.get("conf", 0.50)
            
            inputs = IMG_PROCESSOR(images=pil_image, text=text_query, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = IMG_MODEL(**inputs)
            
            # Post-procesamiento nativo de SAM 3 para instancias por texto
            results = IMG_PROCESSOR.post_process_instance_segmentation(
                outputs,
                threshold=conf_thresh,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            raw_masks = results['masks'].cpu().numpy()
            raw_scores = results['scores'].cpu().numpy()
            
            for idx, mask_array in enumerate(raw_masks):
                output_masks.append({
                    "type": "text_prompt",
                    "label": f"{text_query}",
                    "confidence": float(raw_scores[idx]),
                    "mask_base64": mask_to_base64(mask_array)
                })

        # ==========================================
        # CASO B: Segmentación por CLICS (Coordenadas)
        # ==========================================
        if points_query:
            print(f"Modo: Segmentación por Clics -> Puntos: {points_query}", flush=True)
            # Para clics positivos el label estándar es 1 por cada punto enviado
            labels_query = input_data.get("labels", [1] * len(points_query))
            
            # Formato requerido por el TrackerProcessor de SAM 3: [Batch, Group, Point_Idx, Coord]
            input_points = [[points_query]]
            input_labels = [[labels_query]]
            
            inputs = TRK_PROCESSOR(
                images=pil_image, 
                input_points=input_points, 
                input_labels=input_labels, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = TRK_MODEL(**inputs, multimask_output=False)
            
            # Post-procesamiento nativo para obtener la máscara final binaria
            masks = TRK_PROCESSOR.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"].cpu(), 
                binarize=True
            )[0]
            
            # Extraemos la máscara del objeto trackeado [H, W]
            mask_array = masks[0][0].numpy()
            
            output_masks.append({
                "type": "click_prompt",
                "label": "selected_object",
                "confidence": 1.0, # El Tracker asume el objeto clickeado directamente
                "mask_base64": mask_to_base64(mask_array)
            })

        print(f"Trabajo finalizado. Máscaras generadas: {len(output_masks)}", flush=True)
        return {
            "status": "success",
            "count": len(output_masks),
            "results": output_masks
        }

    except Exception as e:
        print(f"ERROR DURANTE EL PROCESAMIENTO: {str(e)}", flush=True)
        return {"error": str(e)}

# -------------------------
# Inicialización del Servicio Serverless
# -------------------------
runpod.serverless.start({"handler": handler})
