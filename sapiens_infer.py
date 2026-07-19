"""Minimal, VRAM-safe inference harness for Meta Sapiens-0.6B TorchScript models.

Tasks: body-part segmentation, 2D pose (Goliath 308 kpts), depth, surface normals.

Design notes
------------
- All Sapiens checkpoints share the same preprocessing: RGB, resize to
  1024x768 (H x W, bilinear, aspect ratio NOT preserved), /255, then
  ImageNet mean/std normalization -> NCHW.
- An RTX 3070 has only 8 GB VRAM, and a single 0.6B fp32 model is ~2.4 GB.
  Four of them + a detector will not co-reside, so the ModelManager keeps
  every model in CPU RAM and shuttles exactly ONE onto the GPU at a time.
- Pose is top-down: a torchvision Faster R-CNN finds person boxes, each box
  is cropped + resized and fed to the pose model, whose heatmaps we decode.
"""
from __future__ import annotations
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from download_models import resolve as resolve_checkpoint
from goliath_consts import GOLIATH_KEYPOINTS, GOLIATH_SKELETON_INFO, GOLIATH_KPTS_COLORS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing constants (shared by all four tasks)
INPUT_H, INPUT_W = 1024, 768
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# --- Segmentation: 28 Goliath body-part classes -----------------------------
SEG_CLASSES = [
    "Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand",
    "Left Lower Arm", "Left Lower Leg", "Left Shoe", "Left Sock",
    "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot",
    "Right Hand", "Right Lower Arm", "Right Lower Leg", "Right Shoe",
    "Right Sock", "Right Upper Arm", "Right Upper Leg", "Torso",
    "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth",
    "Tongue",
]


def _seg_palette() -> np.ndarray:
    """Deterministic palette: index 0 = gray background, rest seeded-random (RGB)."""
    rng = np.random.RandomState(11)
    colors = rng.randint(0, 255, (len(SEG_CLASSES) - 1, 3))
    colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)
    return colors  # (28, 3) RGB


SEG_PALETTE = _seg_palette()
KPT_NAME_TO_IDX = {name: i for i, name in enumerate(GOLIATH_KEYPOINTS)}


# ---------------------------------------------------------------------------
# Model management (single GPU slot, everything else parked in CPU RAM)
# ---------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self._models: dict[str, torch.jit.ScriptModule] = {}
        self._detector = None
        self._gpu_key: str | None = None

    def _load_sapiens(self, task: str):
        if task not in self._models:
            path = resolve_checkpoint(task)
            print(f"[load] {task}: {path}", flush=True)
            self._models[task] = torch.jit.load(path, map_location="cpu").eval()
        return self._models[task]

    def _load_detector(self):
        if self._detector is None:
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
            print("[load] detector: fasterrcnn_resnet50_fpn_v2", flush=True)
            w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._detector = fasterrcnn_resnet50_fpn_v2(weights=w).eval()
        return self._detector

    def _evict(self):
        """Move whatever is on the GPU back to CPU and free VRAM."""
        if self._gpu_key is None or DEVICE == "cpu":
            self._gpu_key = None
            return
        obj = self._detector if self._gpu_key == "det" else self._models.get(self._gpu_key)
        if obj is not None:
            obj.to("cpu")
        torch.cuda.empty_cache()
        self._gpu_key = None

    def on_gpu(self, key: str):
        """Return the requested model, resident on DEVICE, evicting any other."""
        if self._gpu_key == key:
            return self._detector if key == "det" else self._models[key]
        self._evict()
        obj = self._load_detector() if key == "det" else self._load_sapiens(key)
        obj.to(DEVICE)
        self._gpu_key = key
        return obj


MANAGER = ModelManager()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """rgb: HxWx3 uint8 -> normalized [1,3,1024,768] float tensor (on CPU)."""
    resized = cv2.resize(np.ascontiguousarray(rgb), (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (t - _MEAN) / _STD


@torch.inference_mode()
def _run(task: str, rgb: np.ndarray) -> torch.Tensor:
    model = MANAGER.on_gpu(task)
    x = preprocess(rgb).to(DEVICE)
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.float().cpu()


# ---------------------------------------------------------------------------
# Task postprocessors -> each returns an HxWx3 uint8 RGB image
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_segmentation(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    logits = _run("seg", rgb)                       # [1,28,1024,768]
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    seg = logits.argmax(dim=1)[0].numpy().astype(np.int32)   # (h,w)
    color = SEG_PALETTE[seg]                         # (h,w,3) RGB
    blended = (0.5 * rgb + 0.5 * color).astype(np.uint8)
    # keep background pixels as the original image for a cleaner look
    blended[seg == 0] = rgb[seg == 0]
    return blended


@torch.inference_mode()
def run_depth(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = _run("depth", rgb)                         # [1,1,1024,768]
    depth = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    depth = depth[0, 0].numpy()
    mask = depth > 0
    if mask.sum() > 0:
        dmin, dmax = depth[mask].min(), depth[mask].max()
    else:
        dmin, dmax = depth.min(), depth.max()
    norm = 1.0 - (depth - dmin) / (dmax - dmin + 1e-6)
    norm = np.clip(norm, 0, 1)
    vis = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)   # BGR
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color[~mask] = 128
    return color


@torch.inference_mode()
def run_normal(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = _run("normal", rgb)                        # [1,3,1024,768]
    nm = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    nm = nm[0].numpy().transpose(1, 2, 0)            # (h,w,3)
    norm = np.linalg.norm(nm, axis=-1, keepdims=True)
    unit = nm / (norm + 1e-5)
    vis = (((unit + 1.0) / 2.0) * 255).astype(np.uint8)   # RGB
    return vis


# ---------------------------------------------------------------------------
# Pose (top-down: detector -> per-box heatmap decode -> skeleton drawing)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _detect_people(rgb: np.ndarray, conf: float = 0.4) -> list[tuple[int, int, int, int]]:
    det = MANAGER.on_gpu("det")
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    pred = det(t)[0]
    boxes, labels, scores = pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
    h, w = rgb.shape[:2]
    out = []
    for b, l, s in zip(boxes, labels, scores):
        if int(l) == 1 and float(s) >= conf:         # torchvision COCO: person == 1
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 4 and y2 - y1 > 4:
                out.append((x1, y1, x2, y2))
    return out


@torch.inference_mode()
def _pose_on_box(rgb: np.ndarray, box) -> list[tuple[int, int, float]]:
    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2]
    bw, bh = x2 - x1, y2 - y1
    heat = _run("pose", crop)                         # [1,308,256,192]
    heat = heat[0].numpy()
    hh, hw = heat.shape[1], heat.shape[2]
    kpts = []
    for i in range(heat.shape[0]):
        idx = np.argmax(heat[i])
        y, x = np.unravel_index(idx, (hh, hw))
        conf = float(heat[i].max())
        xi = int(x * bw / hw) + x1
        yi = int(y * bh / hh) + y1
        kpts.append((xi, yi, conf))
    return kpts


def _draw_pose(rgb: np.ndarray, people_kpts, thr: float = 0.5) -> np.ndarray:
    canvas = rgb.copy()
    for kpts in people_kpts:
        # skeleton links first
        for info in GOLIATH_SKELETON_INFO.values():
            a, b = info["link"]
            ia, ib = KPT_NAME_TO_IDX.get(a), KPT_NAME_TO_IDX.get(b)
            if ia is None or ib is None:
                continue
            xa, ya, ca = kpts[ia]
            xb, yb, cb = kpts[ib]
            if ca > thr and cb > thr:
                col = tuple(int(c) for c in info["color"])
                cv2.line(canvas, (xa, ya), (xb, yb), col, 2, cv2.LINE_AA)
        # keypoints on top
        for i, (x, y, c) in enumerate(kpts):
            if c > thr:
                col = tuple(int(v) for v in GOLIATH_KPTS_COLORS[i])
                cv2.circle(canvas, (x, y), 2, col, -1, cv2.LINE_AA)
    return canvas


@torch.inference_mode()
def run_pose(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    boxes = _detect_people(rgb)
    if not boxes:                                     # whole-frame fallback
        boxes = [(0, 0, w, h)]
    people = [_pose_on_box(rgb, b) for b in boxes]
    return _draw_pose(rgb, people)


TASKS = {
    "seg": run_segmentation,
    "pose": run_pose,
    "depth": run_depth,
    "normal": run_normal,
}


def infer(rgb: np.ndarray, tasks: list[str]) -> dict:
    """Run the requested tasks. Returns {task: (rgb_result, elapsed_ms)}."""
    results = {}
    for task in tasks:
        if task not in TASKS:
            continue
        t0 = time.time()
        results[task] = (TASKS[task](rgb), int((time.time() - t0) * 1000))
    return results


def gpu_name() -> str:
    if DEVICE == "cuda":
        return torch.cuda.get_device_name(0)
    return ""
