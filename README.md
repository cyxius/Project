# Weather Robustness Project

SS 2026, Sihan Chen & Yuxuan Chen

We tested how bad weather affects YOLOv8m on BDD100K, and whether MSRCR/CLAHE
preprocessing can recover the lost accuracy without retraining.

## Setup

Python 3.11 + PyTorch 2.6. GPU helps a lot (we used an RTX 3070 laptop).

```
pip install torch ultralytics opencv-python matplotlib pyyaml numpy scipy
```

You also need BDD100K from OpenDataLab (DSDL format). Put the images and annotation
JSONs under `data/`. Not included here - too big.

## Usage

```
# phase 1 - baseline
python scripts/run_phase1.py --phase 1 --skip-download --subset 2000

# phase 2 - preprocessing
python scripts/run_phase1.py --phase 2 --skip-download

# run tests
PYTHONPATH=. python tests/test_metrics.py
```

## What we got

Phase 1: Clean mAP@0.5 = 0.3539, Adverse = 0.2941. That's a 16.9% drop.

Phase 2: CLAHE brought adverse up to 0.3006 (recovers ~11% of the gap).
MSRCR didn't really help (0.2824, actually slightly worse).

Per-weather: snow is the hardest (0.315 mAP), rain is easier (0.344).
Fog only had 17 images so we can't say much about it.

Most of the degradation comes from small objects - bicycles and motorcycles
drop the most. Cars and people are pretty robust.
