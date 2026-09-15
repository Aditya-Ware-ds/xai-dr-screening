"""Torch device selection."""


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
