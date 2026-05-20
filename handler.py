import runpod
import torch
import base64
import io
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

# 1. INICIALIZACIÓN FUERA DEL HANDLER
# Todo lo que esté aquí se ejecuta al levantar el contenedor (Cold Start).
# Mantiene el modelo cargado en la GPU de forma persistente.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cargando SAM 3 en {device}...")

model_id = "facebook/sam3"
processor = Sam3Processor.from_pretrained(model_id)
model = Sam3Model.from_pretrained(model_id).to(device)

def decode_base64_image(image_string):
    """Convierte un string base64 en un objeto PIL Image RGB."""
    if "," in image_string:
        image_string = image_string.split(",")[1]
    image_bytes = base64.b64decode(image_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# 2. FUNCIÓN HANDLER
def handler(job):
    """Función ejecutada por cada petición (job) al endpoint."""
    job_input = job["input"]
    
    image_b64 = job_input.get("image")
    text_prompt = job_input.get("prompt", None)
    
    if not image_b64 or not text_prompt:
        return {"error": "Se requieren los campos 'image' (base64) y 'prompt' (texto) en el input."}
    
    try:
        # Decodificar imagen
        image = decode_base64_image(image_b64)
        
        # Preparar los tensores de entrada
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-procesamiento
        original_size = [image.size[::-1]] # (height, width)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=original_size
        )[0]
        
        # Convertir resultados a formatos serializables (JSON)
        # NOTA: En producción, devolver arrays binarios gigantes puede saturar la red.
        # Lo ideal es comprimir las máscaras (RLE) o devolver coordenadas de contornos.
        # Aquí devolvemos los bounding boxes y el número de objetos encontrados.
        
        scores = results.get("scores", torch.tensor([])).cpu().tolist()
        
        return {
            "status": "success",
            "objects_found": len(scores),
            "scores": scores,
            "message": "Inferencia completada con éxito. Implementa compresión RLE para devolver los mapas de píxeles."
        }
        
    except Exception as e:
        return {"error": str(e)}

# 3. ARRANCAR EL SERVIDOR
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
