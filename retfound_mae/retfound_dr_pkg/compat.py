"""Loads the downloaded models_vit.py as a module and patches it for compatibility
with modern timm versions - it was written against timm~=0.9.2."""
import importlib.util
from pathlib import Path


def load_patched_models_vit(path: Path):
    spec = importlib.util.spec_from_file_location("models_vit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch(module)
    return module


def _patch(module) -> None:
    # Newer timm versions changed VisionTransformer.forward()'s internal call to
    # forward_features(x, attn_mask=..., is_causal=...), but this repo's custom
    # VisionTransformer class (written against timm~=0.9.2) only accepts (self, x).
    # Patch it to silently ignore any extra args/kwargs newer timm passes in -
    # the underlying computation is untouched, this only fixes the call signature.
    orig_forward_features = module.VisionTransformer.forward_features

    def _compat_forward_features(self, x, *args, **kwargs):
        return orig_forward_features(self, x)

    module.VisionTransformer.forward_features = _compat_forward_features

    # models_vit.VisionTransformer.forward_features() (RETFound_mae) already does
    # its own pooling internally: global_pool=True -> mean-pool + fc_norm, giving
    # an already-pooled [B, 1, D] (or [B, D]) tensor; global_pool=False -> norm +
    # cls-token, giving [B, D]. The old timm this was written against then just
    # did head(forward_features(x)). Modern timm's forward() instead calls
    # forward_head() -> pool(), which tries to use self.global_pool (a bool here,
    # e.g. True) as a pool-type *string* like 'avg'/'token' and raises
    # `AssertionError: Unknown pool type True`. Bypass that re-pooling entirely
    # and go straight to the head, as the original code intends.
    def _compat_forward(self, x, *args, **kwargs):
        x = self.forward_features(x)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.head_drop(x)
        return self.head(x)

    module.VisionTransformer.forward = _compat_forward
