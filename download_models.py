"""Download the Sapiens TorchScript checkpoints from HuggingFace.

Run once before starting the app:  python download_models.py
Files are cached in the standard HF cache; MODELS holds the resolved paths.

The normals model size is selectable (0.6b / 1b / 2b) via the SAPIENS_NORMAL_SIZE
environment variable (default 0.6b). `python download_models.py --all-normal`
fetches every normals size so you can switch between them without re-downloading.
"""
import os
import sys
from huggingface_hub import hf_hub_download

# Available normals checkpoints by size -> (repo_id, filename)
NORMAL_CHECKPOINTS = {
    "0.6b": ("facebook/sapiens-normal-0.6b-torchscript",
             "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2"),
    "1b":   ("facebook/sapiens-normal-1b-torchscript",
             "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"),
    "2b":   ("facebook/sapiens-normal-2b-torchscript",
             "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2"),
}

NORMAL_SIZE = os.environ.get("SAPIENS_NORMAL_SIZE", "0.6b").lower()
if NORMAL_SIZE not in NORMAL_CHECKPOINTS:
    raise SystemExit(f"SAPIENS_NORMAL_SIZE must be one of {list(NORMAL_CHECKPOINTS)}, got {NORMAL_SIZE!r}")

# task -> (repo_id, filename). `normal` resolves to the size chosen above.
MODELS = {
    "seg":    ("facebook/sapiens-seg-0.6b-torchscript",
               "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"),
    "pose":   ("facebook/sapiens-pose-0.6b-torchscript",
               "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"),
    "depth":  ("facebook/sapiens-depth-0.6b-torchscript",
               "sapiens_0.6b_render_people_epoch_70_torchscript.pt2"),
    "normal": NORMAL_CHECKPOINTS[NORMAL_SIZE],
}


def resolve(task: str) -> str:
    repo_id, filename = MODELS[task]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def resolve_normal(size: str) -> str:
    repo_id, filename = NORMAL_CHECKPOINTS[size]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def is_normal_cached(size: str) -> bool:
    """True if the normals checkpoint for `size` is already in the local HF cache."""
    repo_id, filename = NORMAL_CHECKPOINTS[size]
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        return True
    except Exception:
        return False


def available_normal_sizes() -> list:
    """Normals sizes present locally (so the UI only offers ones that won't
    trigger a multi-GB download mid-request)."""
    return [s for s in NORMAL_CHECKPOINTS if is_normal_cached(s)]


def download_all(all_normal: bool = False) -> dict:
    paths = {}
    for task in MODELS:
        print(f"[{task}] downloading {MODELS[task][1]} ...", flush=True)
        paths[task] = resolve(task)
        print(f"[{task}] -> {paths[task]}", flush=True)
    if all_normal:
        for size in NORMAL_CHECKPOINTS:
            print(f"[normal:{size}] downloading {NORMAL_CHECKPOINTS[size][1]} ...", flush=True)
            resolve_normal(size)
    return paths


if __name__ == "__main__":
    download_all(all_normal="--all-normal" in sys.argv)
    print("\nAll requested Sapiens checkpoints are cached and ready.")
