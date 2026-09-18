#!/usr/bin/env python3
"""Evaluate the RETFound-MAE DR grader against a labeled dataset.

Prompts for a CSV of ground-truth labels and a directory of fundus images,
runs every image through RetfoundDRScorer, collapses the 5-class ICDR stage
(0-4) to a binary "referable DR" call (stage >= 2 positive, the standard
clinical screening threshold), and reports sensitivity, specificity, and AUC
for that binary task, plus overall 5-class accuracy.

CSV requirements: a header row with a filename column (one of "filename",
"image", "image_id", "file", "path") and a label column (one of "label",
"diagnosis", "stage", "grade") holding the integer ICDR stage 0-4.

Usage:
    python test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "retfound_mae"))

from retfound_dr_pkg import RetfoundDRScorer  # noqa: E402
from retfound_dr_pkg.config import DR_LABELS  # noqa: E402

REFERABLE_THRESHOLD = 2  # ICDR stage >= 2 (Moderate NPDR or worse) counts as positive/referable
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

FILENAME_COLUMNS = ["id_code", "filename", "image", "image_id", "file", "path"]
LABEL_COLUMNS = ["label", "diagnosis", "stage", "grade"]


def clean_path(raw: str) -> Path:
    return Path(raw.strip().strip('"').strip("'")).expanduser()


def prompt_for_path(prompt: str, must_be_file: bool) -> Path:
    while True:
        raw = input(prompt).strip()
        path = clean_path(raw)
        if must_be_file and not path.is_file():
            print(f"  -> No file found at: {path}\n")
            continue
        if not must_be_file and not path.is_dir():
            print(f"  -> No directory found at: {path}\n")
            continue
        return path


def pick_column(columns: list[str], candidates: list[str], kind: str) -> str:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        f"Could not find a {kind} column in the CSV. "
        f"Expected one of {candidates}, found columns: {columns}"
    )


def resolve_image_path(image_dir: Path, filename: str) -> Path | None:
    candidate = image_dir / filename
    if candidate.is_file():
        return candidate
    if not candidate.suffix:
        for ext in IMAGE_EXTENSIONS:
            with_ext = image_dir / f"{filename}{ext}"
            if with_ext.is_file():
                return with_ext
    return None


def main() -> None:
    print("=== RETFound-MAE DR Grader: Test / Evaluation ===\n")

    csv_path = prompt_for_path("Enter path to ground-truth CSV file: ", must_be_file=True)
    image_dir = prompt_for_path("Enter path to directory of fundus images: ", must_be_file=False)

    df = pd.read_csv(csv_path)
    filename_col = pick_column(list(df.columns), FILENAME_COLUMNS, "id_code")
    label_col = pick_column(list(df.columns), LABEL_COLUMNS, "diagnosis")
    print(f"Using '{filename_col}' as filename column and '{label_col}' as label column.\n")

    scorer = RetfoundDRScorer()

    y_true_binary: list[int] = []
    y_score_binary: list[float] = []
    y_pred_binary: list[int] = []
    y_true_stage: list[int] = []
    y_pred_stage: list[int] = []
    skipped: list[str] = []

    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        filename = str(getattr(row, filename_col))
        try:
            true_stage = int(getattr(row, label_col))
        except (TypeError, ValueError):
            skipped.append(f"{filename} (unparseable label: {getattr(row, label_col)!r})")
            continue

        image_path = resolve_image_path(image_dir, filename)
        if image_path is None:
            skipped.append(f"{filename} (image not found in {image_dir})")
            continue

        try:
            result = scorer.score(image_path)
        except Exception as e:
            skipped.append(f"{filename} (inference error: {e})")
            continue

        pred_stage = DR_LABELS.index(result["predicted_stage"])
        positive_score = sum(
            prob for label, prob in result["probabilities"].items()
            if DR_LABELS.index(label) >= REFERABLE_THRESHOLD
        )

        y_true_stage.append(true_stage)
        y_pred_stage.append(pred_stage)
        y_true_binary.append(int(true_stage >= REFERABLE_THRESHOLD))
        y_pred_binary.append(int(pred_stage >= REFERABLE_THRESHOLD))
        y_score_binary.append(positive_score)

        if i % 10 == 0 or i == total:
            print(f"  processed {i}/{total} images...")

    print()
    if skipped:
        print(f"Skipped {len(skipped)} row(s):")
        for reason in skipped:
            print(f"  - {reason}")
        print()

    n = len(y_true_binary)
    if n == 0:
        print("No images were successfully scored. Nothing to evaluate.")
        return

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    if len(set(y_true_binary)) < 2:
        auc = float("nan")
        auc_note = " (undefined: only one class present in ground truth)"
    else:
        auc = roc_auc_score(y_true_binary, y_score_binary)
        auc_note = ""

    stage_accuracy = accuracy_score(y_true_stage, y_pred_stage)
    binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)

    print("=== Results ===")
    print(f"Images evaluated: {n}")
    print(f"Referable-DR definition: ICDR stage >= {REFERABLE_THRESHOLD} = positive\n")
    print(f"Confusion matrix (binary, referable DR):")
    print(f"  TP={tp}  FN={fn}")
    print(f"  FP={fp}  TN={tn}\n")
    print(f"Sensitivity (recall, TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR):         {specificity:.4f}")
    print(f"AUC:                       {auc:.4f}{auc_note}")
    print(f"Binary accuracy:           {binary_accuracy:.4f}")
    print(f"5-class (ICDR 0-4) accuracy: {stage_accuracy:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
