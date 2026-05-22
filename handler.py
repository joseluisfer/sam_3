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
login(token=os.environ["HF_TOKEN"])
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

    return img

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
        # TEXT + IMAGE PROCESSING
        # -------------------------------------------------
        inputs = processor(
            images=img,
            text=text,
            return_tensors="pt"
        )

        # mover a GPU + FP32 FIX
        for k in inputs:
            if torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(DEVICE).float()

        with torch.no_grad():

            outputs = model(**inputs)

        return {
            "status": "success",
            "message": "SAM3 inference done",
            "keys": list(outputs.keys())
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
