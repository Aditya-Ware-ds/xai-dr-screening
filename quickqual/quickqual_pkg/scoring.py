"""Turning backbone features into quality scores and human-readable labels."""
from .config import MEME_BIAS, MEME_FEATURE_IDX, MEME_WEIGHTS


def score_svm(feats, clf) -> dict:
    probs = clf.predict_proba(feats.reshape(1, -1).numpy())[0]
    return {"p_good": float(probs[0]), "p_usable": float(probs[1]), "p_bad": float(probs[2])}


def score_meme(feats) -> dict:
    import torch
    w = torch.tensor(MEME_WEIGHTS)
    b = torch.tensor(MEME_BIAS)
    sub_feats = feats[MEME_FEATURE_IDX]
    p_bad = torch.sigmoid(sub_feats @ w + b).item()
    return {"p_bad": float(p_bad)}


def classify_quality(scores: dict) -> str:
    """Turn a QuickQualScorer.score() result into a human-readable label."""
    if scores["p_bad"] >= 0.5:
        return "BAD"
    if scores.get("p_usable", 0) >= 0.5:
        return "USABLE (borderline)"
    return "GOOD"
