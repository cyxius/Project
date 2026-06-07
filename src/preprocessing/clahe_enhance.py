from pathlib import Path
import cv2
import numpy as np


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


def enhance_directory(input_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    import os
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    if not files:
        return
    n = 0
    for fname in files:
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue
        out = enhance(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        cv2.imwrite(os.path.join(output_dir, fname), out)
        n += 1
    print(f"CLAHE done: {n} images")
    return n
