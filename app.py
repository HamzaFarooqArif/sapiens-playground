"""FastAPI web app: upload a photo, see Sapiens pose / segmentation / depth / normals."""
from __future__ import annotations
import base64
import io
import json
import time
import threading

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
    """Stream results as newline-delimited JSON, one line per task as its model
    finishes, so each output appears in the UI the moment it is ready."""
    data = await image.read()
    try:
        rgb = _read_rgb(data)                 # decode once, up front (errors -> 400)
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    task_list = [t.strip() for t in tasks.split(",") if t.strip() in S.TASKS]
    if not task_list:
        raise HTTPException(400, "No valid tasks requested.")

    def stream():
        # Serialize across requests: one GPU, one model resident at a time.
        with _LOCK:
            for task in task_list:
                try:
                    t0 = time.time()
                    out = S.TASKS[task](rgb)
                    ms = int((time.time() - t0) * 1000)
                    msg = {"task": task, "image": _to_data_uri(out), "ms": ms}
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    msg = {"task": task, "error": str(e)}
                yield json.dumps(msg) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
