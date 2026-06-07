# Enhancing Robustness in Adverse Weather Conditions

Laborprojekt Servicerobotik / Bildverarbeitung fur Robotik — SS 2026

## What we did

We evaluated how bad weather (rain, snow, fog) hurts YOLOv8m object detection on the
BDD100K dataset, and tried to fix it with simple image preprocessing — no model
retraining needed.

- **Phase 1**: Ran YOLOv8m on 2000 clean vs 2000 adverse images, measured the gap.
  Clean mAP@0.5 = 0.3539, Adverse = 0.2941 → **16.9% drop**.
- **Phase 2**: Applied MSRCR and CLAHE to adverse images before feeding them to the
  same model. CLAHE recovered about **10.9%** of the gap. MSRCR didn't help much.

## Setup

Python 3.11, PyTorch 2.6, Ultralytics 8.4. GPU with 8GB VRAM recommended.

```bash
pip install torch ultralytics opencv-python numpy scipy matplotlib pyyaml pillow python-docx
```

## Dataset

We use BDD100K from [OpenDataLab](https://opendatalab.com/BDD100K) (DSDL format).

After downloading, put:
- Images under `data/raw/bdd100k/images/100k/{train,val}/`
- DSDL annotation JSONs under `data/BDD100K/dsdl/det_full/dsdl_Det_full/`
  (these contain weather labels, not included here due to size)

The dataset splits images by weather:
- **Clean**: clear, sunny, partly cloudy, overcast
- **Adverse**: rainy, snowy, foggy

## How to run

```bash
# Phase 1 — baseline
python scripts/run_phase1.py --phase 1 --skip-download --subset 2000

# Phase 2 — preprocessing recovery (full)
python scripts/run_phase1.py --phase 2 --skip-download

# Skip MSRCR/CLAHE if already processed
python scripts/run_phase1.py --phase 2 --skip-download --skip-preprocessing

# Tests
PYTHONPATH=. python tests/test_metrics.py
```

## Results

### Phase 1 — Clean vs Adverse

| | Clean | Adverse | Drop |
|---|-------|---------|------|
| mAP@0.5 | 0.3539 | 0.2941 | 16.9% |

### Phase 2 — Preprocessing Recovery

| Condition | mAP@0.5 | Recovery |
|-----------|---------|----------|
| Clean | 0.3539 | — |
| Adverse (raw) | 0.2941 | baseline |
| + MSRCR | 0.2824 | -19.6% (slight decrease) |
| + CLAHE | 0.3006 | +10.9% |

### Per-Weather

| Weather | Images | mAP@0.5 |
|---------|--------|---------|
| snowy | 992 | 0.315 |
| rainy | 991 | 0.344 |
| foggy | 17 | 0.371* |

*only 17 images, not reliable

## What's next

- RT-DETR (Transformer) — cross-architecture comparison
- Grad-CAM — visualize what the model actually looks at in bad weather

## Team

Sihan Chen (3872154), Yuxuan Chen (3619935)
