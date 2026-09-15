"""Eval-time image preprocessing, copied exactly from util/datasets.py build_transform():
crop_pct = 224/256 for input_size <= 224 -> resize to 256, then center-crop to 224."""
from torchvision import transforms

from .config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_eval_transform():
    """Resize -> crop -> tensor -> normalize. What the model actually sees."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_display_transform():
    """Same resize+crop without ToTensor/Normalize - pixel-aligned with the model's input, for display."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
    ])
