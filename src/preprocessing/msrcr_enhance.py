import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def msrcr(image, sigma_list=None, alpha=125.0, beta=46.0):
    # MSRCR - multi-scale retinex with color restoration
    # sigma_list: gaussian blur scales (tuned these by trial and error)
    # alpha/beta: color restoration strength
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


def enhance_directory(input_dir, output_dir, sigma_list=None, alpha=125.0, beta=46.0):
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jpg"))
    n = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out = msrcr(img, sigma_list=sigma_list, alpha=alpha, beta=beta)
        cv2.imwrite(str(output_dir / p.name), out)
        n += 1
    logger.info("MSRCR: %d images", n)
    return n
