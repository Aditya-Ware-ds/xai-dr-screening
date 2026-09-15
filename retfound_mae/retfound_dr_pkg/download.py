"""Fetching the authors' own models_vit.py, so the exact RETFound_mae /
RETFound_dinov2 architectures they use are used here too, rather than
reimplementing them by hand."""
from pathlib import Path

from .config import MODELS_VIT_URL


def ensure_models_vit(dest_path: Path) -> None:
    if dest_path.exists():
        return
    import requests
    print(f"Downloading models_vit.py from the official repo to {dest_path} ...")
    r = requests.get(MODELS_VIT_URL, timeout=60)
    r.raise_for_status()
    dest_path.write_text(r.text, encoding="utf-8")
    print("Done.")
