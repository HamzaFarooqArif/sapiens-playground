# Sapiens Playground

A small local web app for Meta's **[Sapiens](https://github.com/facebookresearch/sapiens)**
human-vision foundation models. Upload a photo of a person and see four outputs:

- **Segmentation** — 28 Goliath body parts
- **Pose** — Goliath 308 keypoints + skeleton
- **Depth** — per-pixel depth (Inferno colormap)
- **Surface normals** — RGB normal map

Runs the **0.6B** TorchScript checkpoints locally on your GPU. Because an 8 GB card
can't hold all four models at once, only one model is resident on the GPU at a time
(the rest wait in CPU RAM), so tasks run sequentially.

## Setup

```powershell
# 1. Create the venv and install deps
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Download the Sapiens-0.6B checkpoints from HuggingFace (~5 GB, one-time)
.\.venv\Scripts\python.exe download_models.py

# 3. Start the app (auto-picks a free port if 8000 is busy)
.\.venv\Scripts\python.exe run.py
```

The console prints the URL, e.g. `http://127.0.0.1:8000` (or the next free port
if 8000 is taken). Open it, drop in a photo, and click **Run**.

Options: `python run.py --port 8080` (preferred port), `--host 0.0.0.0` (LAN
access), `--reload` (dev auto-reload).

The first run of each task also downloads a person detector
(torchvision Faster R-CNN, ~170 MB) used for top-down pose.

## Notes

- **First inference is slow** — checkpoints load from disk into RAM and get JIT-warmed.
  Subsequent runs are much faster.
- Preprocessing follows Sapiens exactly: RGB, resize to 1024×768 (H×W), ImageNet
  normalization. Input aspect ratio is not preserved (this matches the reference).
- Tested on an RTX 3070 (8 GB), Python 3.11, Windows 11.

## Files

| File | Purpose |
|------|---------|
| `run.py` | Launcher — starts the app, auto-selecting a free port |
| `app.py` | FastAPI server + upload API |
| `sapiens_infer.py` | Preprocessing, model manager, the four task postprocessors, pose decoding |
| `download_models.py` | Resolves/downloads the 0.6B checkpoints from HuggingFace |
| `goliath_consts.py` | Goliath keypoint names, colors, and skeleton (from ibaiGorordo/Sapiens-Pytorch-Inference) |
| `static/index.html` | Web UI |
