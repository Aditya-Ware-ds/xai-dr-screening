"""Rendering the Grad-CAM heatmap/overlay. Only imports matplotlib (an optional,
plotting-only dependency) when a plot is actually requested."""
import numpy as np


def make_overlay(display_img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    heatmap_rgb = (cmap(cam)[..., :3] * 255).astype(np.uint8)
    return ((1 - alpha) * display_img + alpha * heatmap_rgb).clip(0, 255).astype(np.uint8)


def plot_gradcam(display_img, cam, overlay, predicted_label, predicted_prob, target_label) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img)
    axes[0].set_title("Input")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (explaining: {target_label})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Predicted: {predicted_label} ({predicted_prob:.1%})")
    plt.tight_layout()
    plt.show()
