"""Download the Sapiens-0.6B TorchScript checkpoints from HuggingFace.

Run once before starting the app:  python download_models.py
Files are cached in the standard HF cache; MODELS holds the resolved paths.
"""
from huggingface_hub import hf_hub_download

# task -> (repo_id, filename)
MODELS = {
    "seg":    ("facebook/sapiens-seg-0.6b-torchscript",
               "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"),
    "pose":   ("facebook/sapiens-pose-0.6b-torchscript",
               "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"),
    "depth":  ("facebook/sapiens-depth-0.6b-torchscript",
               "sapiens_0.6b_render_people_epoch_70_torchscript.pt2"),
    "normal": ("facebook/sapiens-normal-0.6b-torchscript",
               "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2"),
}


def resolve(task: str) -> str:
    repo_id, filename = MODELS[task]
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download_all() -> dict:
    paths = {}
    for task in MODELS:
        print(f"[{task}] downloading {MODELS[task][1]} ...", flush=True)
        paths[task] = resolve(task)
        print(f"[{task}] -> {paths[task]}", flush=True)
    return paths


if __name__ == "__main__":
    download_all()
    print("\nAll Sapiens-0.6B checkpoints are cached and ready.")
