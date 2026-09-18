# XAI DR Screening

Explainable diabetic retinopathy (DR) screening from a single fundus photograph.

The pipeline gates every image through an automated **image-quality check**
before it ever reaches the clinical model, then grades DR severity on the
standard **ICDR 0–4 scale** and explains *why* with a **Vision-Transformer
Grad-CAM** heatmap — so a prediction is never returned without both a quality
guarantee and a visual justification.

```
 fundus image
      │
      ▼
┌─────────────────┐   BAD    ┌──────────────────────────┐
│    QuickQual     ├─────────▶  reject, ask for a better │
│ (quality gate)   │          │  image, retry            │
└────────┬─────────┘          └──────────────────────────┘
         │ GOOD / USABLE
         ▼
┌─────────────────┐
│  RETFound-MAE    │  ── DR stage (0–4) + class probabilities
│ (ViT DR grader)  │  ── Grad-CAM heatmap + overlay (XAI)
└─────────────────┘
```

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model weights](#model-weights)
- [Usage](#usage)
- [Output](#output)
- [Design notes](#design-notes)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Why this exists

Automated DR grading models are only as trustworthy as the images fed into
them — a blurry, poorly illuminated, or badly cropped fundus photo can silently
produce a confident but wrong grade. This project treats that as a first-class
problem rather than an afterthought:

1. **Quality gating.** Every image is screened by
   [QuickQual](https://github.com/justinengelmann/QuickQual) before grading.
   Images flagged `BAD` are rejected up front instead of being fed to the
   downstream model.
2. **Explainability by default.** Every DR prediction ships with a Grad-CAM
   heatmap showing which retinal regions actually drove the model's decision,
   not just a bare class label.

## How it works

### 1. Quality gate — QuickQual

- Backbone: `DenseNet121` (ImageNet-pretrained, via `timm`) used as a frozen
  feature extractor.
- Head: a pretrained SVM classifies the 512-d feature vector into
  `p_good` / `p_usable` / `p_bad`.
- `classify_quality()` maps those probabilities to a single verdict —
  `GOOD`, `USABLE (borderline)`, or `BAD`. Only `BAD` images are rejected; the
  app then re-prompts for a better image until the gate passes.

### 2. DR staging & explanation — RETFound-MAE

- Backbone: [RETFound](https://github.com/rmaphoh/RETFound) — a ViT-Large
  encoder pretrained with masked-autoencoding (MAE) on ~1.6M retinal images —
  fine-tuned with a 5-class linear head for ICDR DR grading.
- The checkpoint's weight shapes are auto-inspected at load time
  (`retfound_dr_pkg/variant.py`) to detect whether it's the MAE variant
  (16×16 patches) or the DINOv2 variant (14×14 patches), and the matching
  architecture is built automatically from the authors' own `models_vit.py`.
- **Grad-CAM for ViTs**: standard Grad-CAM assumes a convolutional spatial
  feature map, which a transformer doesn't have. `retfound_dr_pkg/gradcam.py`
  adapts it by hooking the last transformer block's patch-token activations
  and gradients, average-pooling the gradient per channel (the Grad-CAM
  weighting step, with "spatial location" = patch token instead of conv
  pixel), and bicubic-upsampling the resulting per-patch importance map back
  to image resolution.

### Output classes (ICDR scale)

| Stage | Meaning              |
|:-----:|----------------------|
| 0     | No DR                |
| 1     | Mild NPDR            |
| 2     | Moderate NPDR        |
| 3     | Severe NPDR           |
| 4     | Proliferative DR      |

## Repository layout

```
xai-dr-screening/
├── app.py                          # CLI entry point: quality gate -> DR grading -> Grad-CAM
├── quickqual/
│   ├── quickqual_pkg/               # QuickQual library (backbone, preprocessing, scoring)
│   └── weights/
│       └── quickqual_dn121_512.pkl  # pretrained SVM head (~26 MB, included)
└── retfound_mae/
    ├── retfound_dr_pkg/             # RETFound library (model build, Grad-CAM, viz)
    ├── models_vit.py                # official RETFound architecture (auto-downloaded if missing)
    ├── checkpoint-best.pth          # fine-tuned RETFound-MAE weights (~3.4 GB, NOT in git — see below)
    └── retfound_mae_dr_staging.ipynb # exploratory notebook version of the pipeline
```

## Requirements

- Python 3.10+
- A CUDA GPU is recommended for the RETFound stage but not required — the app
  falls back to CPU automatically (including mid-run, if a Grad-CAM backward
  pass hits an out-of-memory error).

## Installation

```bash
git clone <repo-url>
cd xai-dr-screening
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install torch torchvision timm scikit-learn joblib matplotlib pillow numpy requests pandas
```

> Pin `torch`/`torchvision` to the build matching your CUDA version — see the
> [PyTorch install matrix](https://pytorch.org/get-started/locally/).

## Model weights

| File | Size | Source |
|---|---|---|
| `quickqual/weights/quickqual_dn121_512.pkl` | ~26 MB | Included in this repo; auto-downloaded from the [QuickQual release](https://github.com/justinengelmann/QuickQual/releases) on first run if missing. |
| `retfound_mae/models_vit.py` | small | Included; auto-downloaded from the [official RETFound repo](https://github.com/rmaphoh/RETFound) on first run if missing. |
| `retfound_mae/checkpoint-best.pth` | ~3.4 GB | **Not tracked in git** (too large). Supply your own DR-fine-tuned RETFound-MAE/DINOv2 checkpoint (5-class ICDR head) at this path before running the app. |

The RETFound loader auto-detects MAE vs. DINOv2 from the checkpoint's patch
size, so either variant works as long as it was fine-tuned with
`num_classes=5`.

## Usage

Run the interactive CLI from the repository root:

```bash
python app.py
```

Example session:

```
=== Diabetic Retinopathy Screening ===
Type 'q' at any prompt to quit.

Enter path to fundus image: sample_images/blurry_fundus.jpg

QuickQual result for blurry_fundus.jpg:
  p_good=0.041  p_usable=0.212  p_bad=0.747  -> BAD
Image quality is too poor for reliable DR grading.

Please provide a better quality image path: sample_images/good_fundus.jpg

QuickQual result for good_fundus.jpg:
  p_good=0.911  p_usable=0.076  p_bad=0.013  -> GOOD
Image quality is good - proceeding.

Running RETFound-MAE DR grading on good_fundus.jpg ...

Predicted DR stage: 2 - Moderate NPDR
Class probabilities:
  0 - No DR              3.1%
  1 - Mild NPDR           9.8%
  2 - Moderate NPDR      71.4%
  3 - Severe NPDR        12.9%
  4 - Proliferative DR    2.8%

Grad-CAM visualization saved to: gradcam_outputs/good_fundus_gradcam.png

Check another image? [y/N]:
```

The first call in a session is slower — the DenseNet121 and RETFound
backbones are loaded and cached once, then reused for every subsequent image.

## Output

Each graded image produces a three-panel PNG in `gradcam_outputs/`:
`<image_stem>_gradcam.png`, containing the model's input crop, the raw
Grad-CAM heatmap, and the heatmap overlaid on the original image — so a
predicted stage is always paired with a visual account of which retinal
regions influenced it. The app runs matplotlib in headless (`Agg`) mode, so
no display is required.

## Design notes

- Both `quickqual_pkg` and `retfound_dr_pkg` are self-contained libraries with
  a small facade class (`QuickQualScorer`, `RetfoundDRScorer`) that lazily
  loads and caches its backbone/device/weights on first use — `app.py`
  instantiates each scorer once and reuses it across every image checked in a
  session.
- Only images classified `BAD` by QuickQual are rejected; `USABLE
  (borderline)` images are allowed through with a warning, since QuickQual's
  own three-way split already treats "usable" as fit for downstream use.
- The RETFound loader memory-maps the checkpoint (`mmap=True`) and loads
  weights directly into the model (`assign=True`) to keep peak RAM usage down
  when reading a several-GB training checkpoint.

## Disclaimer

This project is a research/educational prototype, **not a medical device**.
It has not been clinically validated and must not be used to make or inform
real diagnostic or treatment decisions. Always defer to a qualified
ophthalmologist.

## Acknowledgements

- [QuickQual](https://github.com/justinengelmann/QuickQual) — Justin Engelmann et al., retinal image quality assessment.
- [RETFound](https://github.com/rmaphoh/RETFound) — Yukun Zhou et al., a foundation model for generalizable disease detection from retinal images ([*Nature*, 2023](https://www.nature.com/articles/s41586-023-06555-x)).

## License

Distributed under the terms of the [MIT License](LICENSE).
