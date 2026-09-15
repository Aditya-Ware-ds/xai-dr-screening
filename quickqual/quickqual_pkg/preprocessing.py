"""Image preprocessing: black-border crop, square padding, resize."""
import numpy as np
from PIL import Image


def preprocess_img(img: Image.Image, size: int = 512, border_threshold: int = 15) -> Image.Image:
    img = img.convert("RGB")
    try:
        arr = np.array(img).mean(-1)
        mask = arr > border_threshold
        rows, cols = np.where(mask)
        if rows.size == 0 or cols.size == 0:
            raise ValueError("no non-black content found")
        buffer = 20
        left = max(0, cols.min() - buffer)
        right = min(arr.shape[1], cols.max() + buffer)
        top = max(0, rows.min() - buffer)
        bottom = min(arr.shape[0], rows.max() + buffer)
        img = img.crop((left, top, right, bottom))
    except Exception:
        pass

    width, height = img.size
    if width != height:
        from torchvision.transforms import functional as F
        if width > height:
            pad = width - height
            padding = [0, pad // 2, 0, pad - pad // 2]
        else:
            pad = height - width
            padding = [pad // 2, 0, pad - pad // 2, 0]
        img = F.pad(img, padding)

    return img.resize((size, size), resample=Image.LANCZOS)
