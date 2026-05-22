import runpod
import base64
import cv2
import numpy as np
import torch

# Importamos las clases de la librería oficial de sam3
from sam3.build_sam3 import build_sam3
from sam3.predictor import SAM3Predictor

# --- 1. Inicialización Global (Cold Start) ---
print("Iniciando contenedor y cargando SAM 3 en la GPU...", flush=True)

# Detectamos si hay GPU disponible (RunPod siempre debería tenerla)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # Cargamos el "cerebro" (los pesos) en el "esqueleto" (la arquitectura)
    sam_model = build_sam3(checkpoint="/app/weights/sam3.safetensors")
    sam_model.to(device=device)
    
    # Creamos el predictor, que es la herramienta que procesa las imágenes
    predictor = SAM3Predictor(sam_model)
    print("¡Modelo SAM 3 cargado con éxito!", flush=True)
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}", flush=True)


# --- 2. Funciones Auxiliares para Android ---
def decode_base64_to_image(b64_string):
    """Convierte el string Base64 que envía Android a una imagen para PyTorch"""
    # Limpiamos la cabecera si Android envía "data:image/jpeg;base64,..."
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # PyTorch usa colores RGB, pero OpenCV lee en BGR. Hacemos el cambio:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def encode_mask_to_base64(mask_array):
    """Convierte la matriz matemática de la máscara a una imagen PNG transparente"""
    h, w = mask_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Pintamos de blanco (255, 255, 255) con un 60% de opacidad (150) 
    # la zona donde SAM 3 detectó el objeto
    rgba[mask_array > 0] = [255, 255, 255, 150] 
    
    _, buffer = cv2.imencode('.png', rgba)
    return base64.b64encode(buffer).decode('utf-8')


# --- 3. El Handler (El motor que recibe las peticiones HTTP) ---
def handler(job):
    job_input = job.get('input', {})
    image_b64 = job_input.get('image_base64')
    text_prompt = job_input.get('text_prompt')
    
    if not image_b64 or not text_prompt:
        return {"error": "Faltan campos obligatorios: 'image_base64' o 'text_prompt'"}
        
    try:
        # 3.1 Preparamos la imagen
        image_rgb = decode_base64_to_image(image_b64)
        
        # 3.2 Subimos la imagen a la memoria temporal del modelo
        predictor.set_image(image_rgb)
        
        # 3.3 Hacemos la inferencia usando el texto
        masks, scores, boxes = predictor.predict(
            text=text_prompt,
            multimask_output=False # Falso para que devuelva solo la silueta más probable
        )
        
        # 3.4 Formateamos la respuesta
        resultados = []
        
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                # CRÍTICO: Los modelos devuelven tensores de la GPU. 
                # Hay que convertirlos a listas normales de Python (.cpu().numpy().tolist()) 
                # para que RunPod pueda convertirlos a JSON sin estrellarse.
                box = boxes[i].cpu().numpy().tolist() 
                score = float(scores[i].cpu().numpy())
                mask_b64 = encode_mask_to_base64(masks[i].cpu().numpy())
                
                resultados.append({
                    "box": box,           # Array [x_min, y_min, x_max, y_max]
                    "score": score,       # Confianza de la IA (ej. 0.98)
                    "mask_base64": mask_b64 # Imagen lista para superponer en Android
                })
            
        return {
            "status": "success",
            "prompt": text_prompt,
            "detections": resultados
        }
        
    except Exception as e:
        return {"error": f"Fallo en la inferencia: {str(e)}"}


# --- 4. Arranque del Servidor ---
runpod.serverless.start({"handler": handler})
