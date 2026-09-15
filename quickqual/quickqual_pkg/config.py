"""Constants and shared types for the QuickQual pipeline."""
from pathlib import Path
from typing import Literal

SVM_RELEASE_URL = (
    "https://github.com/justinengelmann/QuickQual/releases/download/1.0/"
    "quickqual_dn121_512.pkl"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SVM_PATH = PROJECT_ROOT / "weights" / "quickqual_dn121_512.pkl"

# Feature indices/weights for the fast linear "meme" bad-quality probe.
MEME_FEATURE_IDX = [71, 109, 121, 53, 55, 123, 29, 133, 84]
MEME_WEIGHTS = [-1411.32, 517.09, 342.41, -707.9, 1442.09, -23.25, -541.64, -8.44, 5.44]
MEME_BIAS = 5.18

Method = Literal["svm", "meme"]
