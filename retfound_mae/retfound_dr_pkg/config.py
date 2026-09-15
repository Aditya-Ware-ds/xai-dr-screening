"""Constants and paths for the RETFound DR staging pipeline."""
from pathlib import Path

MODELS_VIT_URL = "https://raw.githubusercontent.com/rmaphoh/RETFound/main/models_vit.py"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_VIT_PATH = PROJECT_ROOT / "models_vit.py"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "checkpoint-best.pth"

NUM_CLASSES = 5
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard ICDR diabetic retinopathy grading scale used by APTOS2019 / IDRiD / MESSIDOR2.
DR_LABELS = [
    "0 - No DR",
    "1 - Mild NPDR",
    "2 - Moderate NPDR",
    "3 - Severe NPDR",
    "4 - Proliferative DR",
]
