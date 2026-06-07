import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def enhance(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def enhance_directory(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jpg"))
    n = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out = enhance(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        cv2.imwrite(str(output_dir / p.name), out)
        n += 1
    logger.info("CLAHE: %d images", n)
    return n
