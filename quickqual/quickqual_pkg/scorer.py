"""High-level facade: QuickQualScorer caches the backbone/device/SVM across calls."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from .backbone import extract_features, load_backbone
from .config import DEFAULT_SVM_PATH, Method
from .device import get_device
from .download import download_svm
from .preprocessing import preprocess_img
from .scoring import score_meme, score_svm


class QuickQualScorer:
    """Scores retinal image quality, caching the backbone/device/SVM across calls."""

    def __init__(self, svm_path: str | Path = DEFAULT_SVM_PATH):
        self.svm_path = Path(svm_path)
        self._model = None
        self._device = None
        self._clf = None

    def _ensure_backbone(self):
        if self._model is None:
            self._device = get_device()
            print(f"Loading DenseNet121 backbone on {self._device} (first call only)...")
            self._model = load_backbone(self._device)
        return self._model, self._device

    def _ensure_svm(self):
        if self._clf is None:
            import joblib
            download_svm(self.svm_path)
            try:
                self._clf = joblib.load(self.svm_path)
            except Exception:
                # Genuinely truncated/interrupted download (rare) - delete and refetch once.
                self.svm_path.unlink(missing_ok=True)
                download_svm(self.svm_path)
                self._clf = joblib.load(self.svm_path)
        return self._clf

    def score(self, image_path: str | Path, method: Method = "svm") -> dict:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"(If this is a Windows path, make sure you used a raw string, e.g. r'{image_path}', "
                f"or forward slashes - plain strings turn backslash sequences like '\\t' into escape characters.)"
            )

        model, device = self._ensure_backbone()
        img = preprocess_img(Image.open(image_path))
        feats = extract_features(model, img, device)

        result = {"image": str(image_path)}
        if method == "svm":
            result.update(score_svm(feats, self._ensure_svm()))
        elif method == "meme":
            result.update(score_meme(feats))
        else:
            raise ValueError(f"Unknown method {method!r}, expected 'svm' or 'meme'")
        return result
