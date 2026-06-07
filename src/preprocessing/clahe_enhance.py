import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def enhance(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE on LAB L-channel. Returns HxWx3 uint8 BGR."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def enhance_directory(
    input_dir: Path,
    output_dir: Path,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> dict:
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        logger.warning("No .jpg images in %s", input_dir)
        return {"image_count": 0, "time_elapsed_sec": 0, "avg_ms_per_image": 0}

    logger.info("CLAHE: processing %d images ...", len(image_files))
    t_start = time.time()

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read %s", img_path)
            continue
        enhanced = enhance(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        cv2.imwrite(str(output_dir / img_path.name), enhanced)

    elapsed = time.time() - t_start
    count = len(image_files)
    avg_ms = (elapsed / count) * 1000 if count > 0 else 0

    logger.info("CLAHE complete: %d images in %.1fs (%.0f ms/image).", count, elapsed, avg_ms)
    return {"image_count": count, "time_elapsed_sec": round(elapsed, 1), "avg_ms_per_image": round(avg_ms, 1)}
