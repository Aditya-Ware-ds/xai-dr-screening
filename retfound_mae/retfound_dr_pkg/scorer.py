"""High-level facade: RetfoundDRScorer caches the model/device across calls."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DEFAULT_CKPT_PATH, DR_LABELS, IMG_SIZE, NUM_CLASSES
from .device import get_device
from .gradcam import compute_gradcam
from .model import build_and_load_model
from .preprocessing import get_display_transform, get_eval_transform
from .viz import make_overlay, plot_gradcam


class RetfoundDRScorer:
    """Scores diabetic retinopathy stage from a fundus image, caching the model/device across calls."""

    def __init__(self, ckpt_path: str | Path = DEFAULT_CKPT_PATH, num_classes: int = NUM_CLASSES):
        self.ckpt_path = Path(ckpt_path)
        self.num_classes = num_classes
        self._model = None
        self._device = None
        self._eval_transform = get_eval_transform()
        self._display_transform = get_display_transform()

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._device

        model = build_and_load_model(self.ckpt_path, self.num_classes)
        model.eval()

        device = get_device()
        if device.type == "cuda":
            try:
                print(f"Moving model to {device}...")
                model.to(device)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                print(
                    f"CUDA out of memory (your GPU reports "
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total). "
                    f"Falling back to CPU - inference will be slower but should still work."
                )
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                model.to(device)
        else:
            model.to(device)

        gc.collect()
        print(f"Ready on {device}.")
        self._model, self._device = model, device
        return self._model, self._device

    def _load_image(self, image_path: Path) -> Image.Image:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"(If this is a Windows path, use a raw string, e.g. r'{image_path}', or forward slashes.)"
            )
        return Image.open(image_path).convert("RGB")

    def score(self, image_path: str | Path) -> dict:
        """Run RETFound DR-grading on a single fundus image."""
        image_path = Path(image_path)
        img = self._load_image(image_path)
        model, device = self._ensure_model()
        tensor = self._eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        predicted_idx = int(max(range(self.num_classes), key=lambda i: probs[i]))
        return {
            "image": str(image_path),
            "predicted_stage": DR_LABELS[predicted_idx],
            "probabilities": {DR_LABELS[i]: round(probs[i], 4) for i in range(self.num_classes)},
        }

    def explain(
        self,
        image_path: str | Path,
        target_class: int | None = None,
        alpha: float = 0.45,
        show: bool = True,
    ) -> dict:
        """
        Run score() plus a Grad-CAM overlay showing which patches of the retina
        drove the prediction. Shows original / heatmap / overlay side by side
        (unless show=False) and returns the same fields as score() plus
        "gradcam" (the raw [IMG_SIZE, IMG_SIZE] heatmap in [0, 1]).
        """
        image_path = Path(image_path)
        img = self._load_image(image_path)
        model, device = self._ensure_model()

        display_img = np.array(self._display_transform(img))
        tensor = self._eval_transform(img).unsqueeze(0).to(device)

        try:
            cam, used_class, logits = compute_gradcam(model, tensor, IMG_SIZE, target_class)
        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory during Grad-CAM backprop - retrying on CPU (will be slower)...")
            torch.cuda.empty_cache()
            model.to("cpu")
            self._device = torch.device("cpu")
            cam, used_class, logits = compute_gradcam(model, tensor.to("cpu"), IMG_SIZE, target_class)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        predicted_idx = int(max(range(self.num_classes), key=lambda i: probs[i]))

        if show:
            overlay = make_overlay(display_img, cam, alpha)
            plot_gradcam(
                display_img, cam, overlay,
                predicted_label=DR_LABELS[predicted_idx], predicted_prob=probs[predicted_idx],
                target_label=DR_LABELS[used_class],
            )

        return {
            "image": str(image_path),
            "predicted_stage": DR_LABELS[predicted_idx],
            "probabilities": {DR_LABELS[i]: round(probs[i], 4) for i in range(self.num_classes)},
            "gradcam_target": DR_LABELS[used_class],
            "gradcam": cam,
        }
