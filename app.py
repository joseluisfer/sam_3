"""
SAM3 RunPod Serverless Handler
==============================
El HF_TOKEN se inyecta como variable de entorno del endpoint en RunPod.
El modelo se descarga la primera vez que arranca el contenedor (cold start).
"""

import os
import runpod
import base64
import numpy as np
import cv2
import torch
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────
# 1. HF_TOKEN — huggingface_hub lo lee automáticamente del entorno.
#    No hace falta llamar a login(). Solo verificamos que esté presente.
# ─────────────────────────────────────────────────────────────
if not os.environ.get("HF_TOKEN"):
    print("[WARN] HF_TOKEN no encontrado. La descarga de modelos privados fallará.", flush=True)
else:
    print("[OK] HF_TOKEN detectado.", flush=True)

# ─────────────────────────────────────────────────────────────
# 2. Carga del modelo (ocurre una sola vez al arrancar)
# ─────────────────────────────────────────────────────────────
print("[SAM3] Cargando modelo...", flush=True)
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("[SAM3] ¡Modelo cargado con éxito!", flush=True)
except Exception as e:
    print(f"[SAM3] Error crítico al cargar el modelo: {e}", flush=True)
    raise


# ─────────────────────────────────────────────────────────────
# 3. Utilidades
# ─────────────────────────────────────────────────────────────
def encode_mask_to_base64(mask_tensor) -> str:
    """
    Convierte una máscara booleana (tensor o ndarray) a PNG con canal alpha,
    codificado en base64. Listo para pintar sobre la imagen en el cliente.
    """
    mask_np = mask_tensor.cpu().numpy() if hasattr(mask_tensor, "cpu") else np.array(mask_tensor)

    # Eliminar dimensiones extra: (1, H, W) → (H, W)
    while len(mask_np.shape) > 2:
        mask_np = mask_np[0]

    h, w = mask_np.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_np > 0] = [255, 255, 255, 150]   # blanco semitransparente

    _, buffer = cv2.imencode(".png", rgba)
    return base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────────────────────
# 4. Handler principal
# ─────────────────────────────────────────────────────────────
def handler(job):
    """
    Entrada esperada (JSON):
    {
        "image_base64": "<base64 con o sin cabecera data:image/...>",
        "text_prompt":  "cat"
    }

    Salida:
    {
        "status":     "success",
        "prompt":     "cat",
        "detections": [
            {
                "score":       0.92,
                "box":         [x1, y1, x2, y2],
                "mask_base64": "<png base64>"
            },
            ...
        ]
    }
    """
    job_input = job.get("input", {})

    # ── Validación ──
    image_b64   = job_input.get("image_base64")
    text_prompt = job_input.get("text_prompt")

    if not image_b64:
        return {"status": "error", "message": "Campo 'image_base64' obligatorio."}
    if not text_prompt:
        return {"status": "error", "message": "Campo 'text_prompt' obligatorio."}

    try:
        # Limpiar cabecera "data:image/jpeg;base64,..." si viene del navegador
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        # Decodificar imagen
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Inferencia SAM3
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

        masks  = output.get("masks",  [])
        boxes  = output.get("boxes",  [])
        scores = output.get("scores", [])

        detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                detections.append({
                    "score":       round(float(scores[i].cpu().numpy()), 4),
                    "box":         [round(v, 2) for v in boxes[i].cpu().numpy().tolist()],
                    "mask_base64": encode_mask_to_base64(masks[i]),
                })

        return {
            "status":     "success",
            "prompt":     text_prompt,
            "detections": detections,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─────────────────────────────────────────────────────────────
# 5. Arranque del servidor serverless
# ─────────────────────────────────────────────────────────────
runpod.serverless.start({"handler": handler})
