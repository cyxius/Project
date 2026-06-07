import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def msrcr(
    image: np.ndarray,
    sigma_list: list[float] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
) -> np.ndarray:
    """MSRCR enhancement."""
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    img_float = image.astype(np.float64) + 1.0
    msr = np.zeros_like(img_float)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
        msr += np.log10(img_float) - np.log10(blurred)
    msr /= len(sigma_list)

    ch_sum = np.sum(img_float, axis=2, keepdims=True)
    cr = beta * (np.log10(alpha * img_float) - np.log10(ch_sum))
    result = cr * msr

    out = np.zeros_like(result, dtype=np.uint8)
    for c in range(3):
        ch = result[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        ch = np.clip((ch - lo) / (hi - lo + 1e-8) * 255.0, 0, 255)
        out[:, :, c] = ch.astype(np.uint8)
    return out


def enhance_directory(
    input_dir: Path,
    output_dir: Path,
    sigma_list: list[float] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
) -> dict:
    """Batch MSRCR enhance all .jpg images in input_dir."""
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        logger.warning("No .jpg images in %s", input_dir)
        return {"image_count": 0, "time_elapsed_sec": 0, "avg_ms_per_image": 0}

    logger.info("MSRCR: processing %d images ...", len(image_files))
    t_start = time.time()

    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read %s", img_path)
            continue
        enhanced = msrcr(image, sigma_list=sigma_list, alpha=alpha, beta=beta)
        cv2.imwrite(str(output_dir / img_path.name), enhanced)

    elapsed = time.time() - t_start
    count = len(image_files)
    avg_ms = (elapsed / count) * 1000 if count > 0 else 0

    logger.info("MSRCR complete: %d images in %.1fs (%.0f ms/image).", count, elapsed, avg_ms)
    return {"image_count": count, "time_elapsed_sec": round(elapsed, 1), "avg_ms_per_image": round(avg_ms, 1)}
