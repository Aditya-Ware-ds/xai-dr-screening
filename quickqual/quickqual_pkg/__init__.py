"""QuickQual retinal image quality screening toolkit.

Wraps the QuickQual DenseNet121 feature extractor with either the pretrained
SVM classifier (good / usable / bad) or a lightweight linear "meme" probe
that flags degenerate/corrupted images without needing the SVM at all.

Typical usage:
    from quickqual_pkg import QuickQualScorer, classify_quality

    scorer = QuickQualScorer()
    scores = scorer.score("path/to/image.png")
    print(scores, classify_quality(scores))
"""
from .scorer import QuickQualScorer
from .scoring import classify_quality

__all__ = ["QuickQualScorer", "classify_quality"]
