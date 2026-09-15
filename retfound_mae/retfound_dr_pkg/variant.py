"""Telling RETFound-MAE and RETFound-DINOv2 checkpoints apart from their weight
shapes alone, so build_and_load_model() builds whichever architecture actually
matches:
  - MAE:    patch_embed.proj.weight is [.., 16, 16], has fc_norm, no LayerScale
  - DINOv2: patch_embed.proj.weight is [.., 14, 14], has blocks.*.ls1/ls2.gamma
"""


def detect_variant(state_dict) -> str:
    w = state_dict.get("patch_embed.proj.weight")
    if w is None:
        raise ValueError("Could not find patch_embed.proj.weight in the checkpoint - unexpected format.")
    patch_size = w.shape[-1]
    if patch_size == 16:
        return "mae"
    elif patch_size == 14:
        return "dinov2"
    raise ValueError(
        f"Unrecognised patch size {patch_size} in checkpoint - doesn't look like a "
        f"RETFound-MAE or RETFound-DINOv2 checkpoint."
    )
