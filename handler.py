import runpod
import torch
import numpy as np
import base64
import cv2
import requests
import os
import sys

from huggingface_hub import login
from transformers import AutoModel, AutoProcessor

# -------------------------------------------------
# Torch config
# -------------------------------------------------
torch.set_default_dtype(torch.float32)

# -------------------------------------------------
# HF Login
# -------------------------------------------------
print("Logging into HuggingFace...", flush=True)

login(token=os.environ["HF_TOKEN"])

print("HF login successful", flush=True)

# -------------------------------------------------
# Load model
# -------------------------------------------------
print("Loading SAM3...", flush=True)

try:

    processor = AutoProcessor.from_pretrained(
        "facebook/sam3",
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        "facebook/sam3",
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).cuda()

    model.eval()

    print("SAM3 loaded successfully.", flush=True)

except Exception as e:

    print(f"MODEL LOAD ERROR: {str(e)}", flush=True)
    sys.exit(1)

# -------------------------------------------------
# Image loader
# -------------------------------------------------
def load_image(data):

    if not isinstance(data, str):
        raise ValueError("Input must be string")

    # URL
    if data.startswith("http"):

        print(f"Downloading: {data[:80]}", flush=True)

        resp = requests.get(data, timeout=20)
        resp.raise_for_status()

        img_array = np.frombuffer(resp.content, np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    else:

        if data.startswith("data:image"):
            data = data.split(",")[1]

        img_bytes = base64.b64decode(data)

        img_array = np.frombuffer(img_bytes, np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    return img

# -------------------------------------------------
# Resize helper
# -------------------------------------------------
def resize_max(img, max_size=1280):

    h, w = img.shape[:2]

    scale = min(max_size / w, max_size / h)

    if scale >= 1:
        return img

    nw = int(w * scale)
    nh = int(h * scale)

    return cv2.resize(img, (nw, nh))

# -------------------------------------------------
# Handler
# -------------------------------------------------
def handler(job):

    try:

        input_data = job["input"]

        print(f"JOB RECEIVED: {job['id']}", flush=True)

        # -------------------------------------------------
        # Required image
        # -------------------------------------------------
        if "file" not in input_data:
            return {"error": "Missing 'file'"}

        img = load_image(input_data["file"])

        img = resize_max(
            img,
            input_data.get("max_size", 1280)
        )

        print(f"Main image shape: {img.shape}", flush=True)

        # -------------------------------------------------
        # Optional pattern
        # -------------------------------------------------
        pattern_img = None

        if input_data.get("pattern"):

            pattern_img = load_image(input_data["pattern"])

            pattern_img = resize_max(pattern_img, 512)

            print(
                f"Pattern shape: {pattern_img.shape}",
                flush=True
            )

        # -------------------------------------------------
        # Optional text
        # -------------------------------------------------
        text_prompt = input_data.get("text", "object")

        print(f"Text prompt: {text_prompt}", flush=True)

        # -------------------------------------------------
        # Processor
        # -------------------------------------------------
        inputs = processor(
            images=img,
            text=text_prompt,
            return_tensors="pt"
        )

        inputs = {
            k: v.float().cuda()
            if torch.is_tensor(v)
            else v
            for k, v in inputs.items()
        }

        print("Running inference...", flush=True)

        with torch.no_grad():

            outputs = model(**inputs)

        print("Inference OK", flush=True)

        # -------------------------------------------------
        # Raw output
        # -------------------------------------------------
        return {
            "status": "success",
            "message": "SAM3 inference completed",
            "output_keys": list(outputs.keys())
        }

    except Exception as e:

        print(f"INFERENCE ERROR: {str(e)}", flush=True)

        return {
            "status": "error",
            "message": str(e)
        }

# -------------------------------------------------
# RunPod
# -------------------------------------------------
runpod.serverless.start({"handler": handler})
