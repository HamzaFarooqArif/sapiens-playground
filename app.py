"""FastAPI web app: upload a photo, see Sapiens pose / segmentation / depth / normals."""
from __future__ import annotations
import base64
import io
import threading

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sapiens_infer as S

app = FastAPI(title="Sapiens Playground")

# One GPU, one model slot -> serialize inference across requests.
_LOCK = threading.Lock()
MAX_SIDE = 1536  # cap huge uploads so drawing / memory stays sane


def _read_rgb(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")   # honor phone orientation
    w, h = img.size
    scale = MAX_SIDE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img)


def _to_data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


@app.get("/api/health")
def health():
    return {"device": S.DEVICE, "gpu": S.gpu_name()}


@app.post("/api/infer")
async def infer(image: UploadFile = File(...), tasks: str = Form("seg,pose,depth,normal")):
    data = await image.read()
    try:
        rgb = _read_rgb(data)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    task_list = [t.strip() for t in tasks.split(",") if t.strip() in S.TASKS]
    if not task_list:
        raise HTTPException(400, "No valid tasks requested.")

    with _LOCK:
        try:
            results = S.infer(rgb, task_list)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Inference failed: {e}")

    return JSONResponse({
        "results": {t: _to_data_uri(img) for t, (img, _ms) in results.items()},
        "timings": {t: ms for t, (_img, ms) in results.items()},
    })


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
