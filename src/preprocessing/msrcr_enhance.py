from pathlib import Path
import cv2
import numpy as np


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


def enhance_directory(input_dir, output_dir, sigma_list=None, alpha=125.0, beta=46.0):
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs = list(input_dir.glob("*.jpg"))
    print(f"MSRCR: {len(imgs)} images")
    done = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is not None:
            cv2.imwrite(str(output_dir / p.name), msrcr(img, sigma_list, alpha, beta))
            done += 1
    print(f"MSRCR done: {done}")
    return done
