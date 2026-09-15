"""ViT-adapted Grad-CAM.

Standard Grad-CAM needs a convolutional spatial feature map, which a Vision
Transformer doesn't have. This instead:

1. Hooks the output of the last transformer block (`model.blocks[-1]`),
   shape [1, num_tokens, D] (cls/register tokens + one token per patch).
2. Backprops the target class logit to get the gradient of that activation.
3. Drops the non-patch (cls/register) tokens, average-pools the gradient
   over the remaining patch tokens to get a per-channel importance weight
   (the Grad-CAM weighting step, with "spatial location" = patch token
   instead of conv pixel), then does a weighted sum over channels per token.
4. ReLU's the result, reshapes the square patch grid, and bicubic-upsamples
   it back to image resolution.

This matches how global_pool=True actually classifies: the head is fed the
mean-pooled patch tokens (the cls token is dropped before pooling), so
explaining via patch tokens - not the cls token - is the right target.
"""
import torch
import torch.nn.functional as F


def compute_gradcam(model, tensor, img_size: int, target_class: int | None = None):
    """
    Returns (cam, target_class, logits) where cam is an [img_size, img_size]
    float32 numpy array in [0, 1] (0 = irrelevant, 1 = most influential).
    """
    target_layer = model.blocks[-1]
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    try:
        tensor = tensor.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        act = activations["value"][0]   # [num_tokens, D]
        grad = gradients["value"][0]    # [num_tokens, D]

        # Non-patch tokens (cls token, and register tokens if the arch has
        # them) sit before the patch tokens - drop however many there are.
        num_patches = model.patch_embed.num_patches
        prefix_len = act.shape[0] - num_patches
        act = act[prefix_len:]    # [num_patches, D]
        grad = grad[prefix_len:]  # [num_patches, D]

        weights = grad.mean(dim=0)                 # [D] - Grad-CAM's per-channel weight
        cam = F.relu((weights * act).sum(dim=1))    # [num_patches]

        grid = int(round(num_patches ** 0.5))
        cam = cam.reshape(1, 1, grid, grid)
        cam = F.interpolate(cam, size=(img_size, img_size), mode="bicubic", align_corners=False)[0, 0]
        cam = cam.clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), target_class, logits.detach()
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
        model.zero_grad(set_to_none=True)
