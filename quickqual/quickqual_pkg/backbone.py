"""DenseNet121 backbone loading and feature extraction."""
from PIL import Image


def load_backbone(device):
    import timm
    model = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
    model.eval().to(device)
    return model


def extract_features(model, img: Image.Image, device):
    import torch
    from torchvision.transforms import functional as F
    tensor = F.to_tensor(img)
    tensor = F.normalize(tensor, [0.5] * 3, [0.5] * 3).to(device).unsqueeze(0)
    with torch.no_grad():
        feats = model(tensor).squeeze().cpu()
    return feats
