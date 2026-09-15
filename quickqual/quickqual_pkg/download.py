"""Fetching the pretrained QuickQual SVM classifier weights."""
from pathlib import Path

from .config import SVM_RELEASE_URL


def download_svm(dest_path: Path) -> None:
    if dest_path.exists():
        return
    import requests
    print(f"Downloading pretrained QuickQual SVM to {dest_path} ...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with requests.get(SVM_RELEASE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp_path.replace(dest_path)
    print("Done.")
