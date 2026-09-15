#!/usr/bin/env python3
"""CLI: quality-gate a fundus image with QuickQual, then grade its diabetic
retinopathy stage and produce a Grad-CAM explanation with RETFound-MAE.

Usage:
    python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: we save figures to disk instead of showing them

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "quickqual"))
sys.path.insert(0, str(ROOT / "retfound_mae"))

from quickqual_pkg import QuickQualScorer, classify_quality  # noqa: E402
from retfound_dr_pkg import RetfoundDRScorer  # noqa: E402
from retfound_dr_pkg.preprocessing import get_display_transform  # noqa: E402
from retfound_dr_pkg.viz import make_overlay  # noqa: E402

GRADCAM_OUTPUT_DIR = ROOT / "gradcam_outputs"


def clean_path(raw: str) -> Path:
    return Path(raw.strip().strip('"').strip("'")).expanduser()


def prompt_for_image_path(prompt: str) -> Path | None:
    """Returns a validated image path, or None if the user asked to quit."""
    while True:
        raw = input(prompt).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        path = clean_path(raw)
        if not path.is_file():
            print(f"  -> No file found at: {path}\n")
            continue
        return path


def check_quality(scorer: QuickQualScorer, image_path: Path) -> bool:
    """Runs QuickQual and reports whether the image is good enough to grade."""
    scores = scorer.score(image_path)
    label = classify_quality(scores)

    print(f"\nQuickQual result for {image_path.name}:")
    print(
        f"  p_good={scores.get('p_good', 0):.3f}  "
        f"p_usable={scores.get('p_usable', 0):.3f}  "
        f"p_bad={scores.get('p_bad', 0):.3f}  -> {label}"
    )

    if label == "BAD":
        print("Image quality is too poor for reliable DR grading.\n")
        return False
    if label.startswith("USABLE"):
        print("Image quality is borderline but usable - proceeding.\n")
    else:
        print("Image quality is good - proceeding.\n")
    return True


def save_gradcam_panel(image_path: Path, result: dict) -> Path:
    """Builds the input/heatmap/overlay panel and saves it next to gradcam_outputs/."""
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    display_img = np.array(get_display_transform()(img))
    cam = result["gradcam"]
    overlay = make_overlay(display_img, cam, alpha=0.45)

    predicted_label = result["predicted_stage"]
    predicted_prob = result["probabilities"][predicted_label]

    GRADCAM_OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = GRADCAM_OUTPUT_DIR / f"{image_path.stem}_gradcam.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (explaining: {result['gradcam_target']})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Predicted: {predicted_label} ({predicted_prob:.1%})")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_dr_grading(scorer: RetfoundDRScorer, image_path: Path) -> None:
    print(f"Running RETFound-MAE DR grading on {image_path.name} ...")
    result = scorer.explain(image_path, show=False)

    print(f"\nPredicted DR stage: {result['predicted_stage']}")
    print("Class probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  {label:<22} {prob:.1%}")

    out_path = save_gradcam_panel(image_path, result)
    print(f"\nGrad-CAM visualization saved to: {out_path}")


def main() -> None:
    print("=== Diabetic Retinopathy Screening ===")
    print("Type 'q' at any prompt to quit.\n")

    quality_scorer = QuickQualScorer()
    dr_scorer = RetfoundDRScorer()

    while True:
        image_path = prompt_for_image_path("Enter path to fundus image: ")
        if image_path is None:
            break

        try:
            while not check_quality(quality_scorer, image_path):
                image_path = prompt_for_image_path("Please provide a better quality image path: ")
                if image_path is None:
                    return
            run_dr_grading(dr_scorer, image_path)
        except FileNotFoundError as e:
            print(f"Error: {e}\n")
        except Exception as e:
            print(f"Unexpected error while processing {image_path}: {e}\n")

        again = input("\nCheck another image? [y/N]: ").strip().lower()
        if again != "y":
            break

    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
