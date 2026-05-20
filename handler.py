import runpod
import numpy as np
import base64
import cv2
import requests
import sys
import torch
import os
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from huggingface_hub import login

print("Iniciando contenedor y verificando token...", flush=True)

try:
    # 1. Leemos el token que pusiste en la interfaz web de RunPod
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("No se encontró la variable HF_TOKEN en RunPod.")
    
    # 2. Iniciamos sesión en Hugging Face
    login(hf_token)
    
# 3. Descargamos/Cargamos el modelo oficial de SAM 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Descargando/Cargando SAM 3 (Hiera Base) en {device}...", flush=True)
    
    # 💥 CAMBIAMOS 'facebook/sam3' POR EL REPOSITORIO REAL 💥
    processor = Sam3Processor.from_pretrained("facebook/sam3-hiera-base")
    model = Sam3Model.from_pretrained("facebook/sam3-hiera-base").to(device)
    
    print("Modelo cargado exitosamente.", flush=True)
except Exception as e:
    print(f"ERROR CRÍTICO AL CARGAR EL MODELO: {str(e)}", flush=True)
    sys.exit(1)

# -------------------------
# Loader de Imágenes
# -------------------------
def load_image(data):
    if not isinstance(data, str):
        raise ValueError("Input must be string")

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
        raise ValueError("Image decode failed (cv2 devolvió None)")

    # Transformers y SAM 3 prefieren RGB en formato PIL nativo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# -------------------------
# Handler Dinámico (Texto / Imagen / Ambos)
# -------------------------
def handler(job):
    try:
        input_data = job["input"]
        print(f"--- NUEVO TRABAJO RECIBIDO (ID: {job.get('id', 'N/A')}) ---", flush=True)

        # 1. El archivo principal SIEMPRE es obligatorio
        if "file" not in input_data:
            return {"error": "Falta el campo obligatorio 'file' (imagen principal)"}

        text_data = input_data.get("text")
        pattern_data = input_data.get("pattern")
        
        # Opcional en SAM 3: soportar cajas directamente [x1, y1, x2, y2]
        box_data = input_data.get("box") 

        # 2. Validar que al menos exista un método de guía
        if not text_data and not pattern_data and not box_data:
            return {"error": "Debes proporcionar al menos un prompt de 'text', un 'pattern' visual, o un 'box'"}

        # 3. Carga de la imagen principal
        print("Cargando imagen principal...", flush=True)
        img = load_image(input_data["file"])

        # 4. Construir argumentos base de la inferencia
        threshold = input_data.get("conf", 0.5)
        processor_args = {
            "images": img,
            "return_tensors": "pt"
        }

        # CASO A: Prompt de Texto
        if text_data:
            print(f"Configurando prompt de texto: {text_data}", flush=True)
            processor_args["text"] = text_data

        # CASO B: Prompt Visual (Imagen de referencia)
        if pattern_data:
            print("Configurando prompt visual (pattern)...", flush=True)
            ref_img = load_image(pattern_data)
            processor_args["reference_images"] = ref_img

        # CASO C: Prompt de Caja (si envían coordenadas crudas)
        if box_data:
            print(f"Configurando caja delimitadora: {box_data}", flush=True)
            processor_args["input_boxes"] = [[box_data]]
            processor_args["input_boxes_labels"] = [[1]] # 1 = positivo

        # 5. Ejecución de la inferencia
        print("Iniciando inferencia en el modelo...", flush=True)
        inputs = processor(**processor_args).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)

        # 6. Post-procesamiento
        # Convierte los tensores crudos en cajas, máscaras y scores legibles
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=[img.size[::-1]] # formato (alto, ancho)
        )[0]
        
        print("Inferencia completada con éxito.", flush=True)

        # 7. Formatear salida (Misma estructura que usabas en YOLOE)
        detections = []
        if "boxes" in results and len(results["boxes"]) > 0:
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                detections.append({
                    "bbox": [round(float(x), 2) for x in box], # [x_min, y_min, x_max, y_max]
                    "confidence": round(float(score), 4),
                    "name": text_data if isinstance(text_data, str) else "object"
                })

        print(f"Trabajo finalizado. Detecciones encontradas: {len(detections)}", flush=True)
        return {
            "status": "success",
            "count": len(detections),
            "detections": detections
        }

    except Exception as e:
        print(f"ERROR DURANTE EL PROCESAMIENTO: {str(e)}", flush=True)
        return {"error": str(e)}

# -------------------------
# RunPod
# -------------------------
runpod.serverless.start({"handler": handler})
