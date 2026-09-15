"""RETFound diabetic retinopathy staging toolkit.

Wraps the RETFound-MAE / RETFound-DINOv2 ViT backbone (auto-detected from the
checkpoint's weight shapes) with a 5-class ICDR DR-stage head and a
ViT-adapted Grad-CAM explainer.

Typical usage:
    from retfound_dr_pkg import RetfoundDRScorer

    scorer = RetfoundDRScorer()
    result = scorer.score("path/to/image.png")
    print(result["predicted_stage"])

    explained = scorer.explain("path/to/image.png")  # also plots the Grad-CAM overlay
"""
from .config import DR_LABELS
from .scorer import RetfoundDRScorer

__all__ = ["RetfoundDRScorer", "DR_LABELS"]
