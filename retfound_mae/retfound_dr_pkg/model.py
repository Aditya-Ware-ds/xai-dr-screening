"""Building and loading the RETFound-MAE / RETFound-DINOv2 backbone from a checkpoint."""
import gc
from pathlib import Path

import torch

from .compat import load_patched_models_vit
from .config import DEFAULT_MODELS_VIT_PATH, IMG_SIZE
from .download import ensure_models_vit
from .variant import detect_variant


def build_and_load_model(ckpt_path: Path, num_classes: int):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download a DINOv2 or MAE checkpoint for your chosen dataset and point ckpt_path at it."
        )

    ensure_models_vit(DEFAULT_MODELS_VIT_PATH)
    models_vit = load_patched_models_vit(DEFAULT_MODELS_VIT_PATH)

    # This checkpoint is a full training checkpoint (weights + optimizer state,
    # ~3x the model's actual size), not just the model weights - on a machine
    # with limited RAM, loading it the naive way can use enough memory to make
    # the whole system thrash/freeze. Two things keep peak memory down:
    #  1. mmap=True (torch>=2.1): the file's tensors are memory-mapped instead
    #     of being read fully into RAM up front - only the "model" key's bytes
    #     actually get paged in when we touch them.
    #  2. assign=True in load_state_dict (torch>=2.1): the checkpoint's tensors
    #     are used directly as the model's parameters instead of allocating a
    #     second full copy of the model's weights to copy the data into.
    # Both are best-effort with a fallback for older torch versions.
    print(f"Reading checkpoint from disk ({ckpt_path.stat().st_size / 1e9:.1f} GB file)...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # Older torch without the mmap kwarg - falls back to reading it all into RAM.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = {k: v for k, v in state_dict.items()}  # detach from the rest of `checkpoint` (optimizer state etc.)
    del checkpoint
    gc.collect()

    variant = detect_variant(state_dict)
    print(f"Detected checkpoint architecture: RETFound-{variant.upper()}")

    print("Building model...")
    if variant == "mae":
        model = models_vit.RETFound_mae(
            img_size=IMG_SIZE, num_classes=num_classes, drop_path_rate=0.2, global_pool=True,
        )
    else:
        # RETFound_dinov2(args, **kwargs) doesn't actually use args - safe to pass None.
        # Note: this pulls Meta's DINOv2 ImageNet weights via timm on first call (one-time
        # download, ~1.2GB) before we immediately overwrite them with the checkpoint below.
        model = models_vit.RETFound_dinov2(None, num_classes=num_classes, drop_path_rate=0.2)

    print("Loading weights into model...")
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        # Older torch without the assign kwarg.
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint looked like RETFound-{variant.upper()} (from its patch size) but still "
                f"didn't load cleanly. Double check num_classes matches how this checkpoint "
                f"was fine-tuned (should be 5 for DR grading).\nOriginal error: {e}"
            )
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint looked like RETFound-{variant.upper()} (from its patch size) but still "
            f"didn't load cleanly. Double check num_classes matches how this checkpoint "
            f"was fine-tuned (should be 5 for DR grading).\nOriginal error: {e}"
        )

    del state_dict
    gc.collect()
    print("Model built and weights loaded.")
    return model
