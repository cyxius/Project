"""Multi-Scale Retinex with Color Restoration (MSRCR) — PyTorch CUDA version."""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pre-compute Gaussian kernels once (module-level cache)
_KERNEL_CACHE = {}


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Build 1D Gaussian kernel [1, 1, 1, k] for separable conv2d."""
    if sigma in _KERNEL_CACHE:
        return _KERNEL_CACHE[sigma].clone()

    radius = int(sigma * truncate + 0.5)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-x**2 / (2.0 * sigma**2))
    kernel /= kernel.sum()
    # Shape: (out_ch, in_ch/groups, kH, kW) for groups=1 with 1 channel
    kernel = kernel.view(1, 1, 1, -1)  # [1, 1, 1, size]
    _KERNEL_CACHE[sigma] = kernel.clone()
    return kernel


def _separable_gaussian_blur(img_gpu: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Separable Gaussian blur on GPU.
    img_gpu: [1, C, H, W] float32 on CUDA.
    """
    kernel_h = _gaussian_kernel_1d(sigma).to(img_gpu.device)  # [1, 1, 1, k]
    kernel_w = kernel_h.transpose(-1, -2)  # [1, 1, k, 1]

    C = img_gpu.shape[1]
    pad_h = (kernel_h.shape[-1] - 1) // 2
    pad_w = (kernel_w.shape[-2] - 1) // 2

    # Horizontal pass (per channel via groups=C)
    x = F.conv2d(img_gpu, kernel_h.expand(C, 1, 1, -1), groups=C, padding=(0, pad_h))
    # Vertical pass
    x = F.conv2d(x, kernel_w.expand(C, 1, -1, 1), groups=C, padding=(pad_w, 0))
    return x


def msrcr(
    image: np.ndarray,
    sigma_list: list[float] = None,
    alpha: float = 125.0,
    beta: float = 46.0,
) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration — GPU-accelerated.

    Args:
        image: HxWx3 uint8 BGR image.
        sigma_list: Gaussian sigmas. Default [15, 80, 250].
        alpha, beta: Color restoration params.

    Returns:
        HxWx3 uint8 BGR image.
    """
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    if not torch.cuda.is_available():
        # CPU fallback
        return _msrcr_cpu(image, sigma_list, alpha, beta)

    device = torch.device("cuda:0")

    # BGR → RGB → float32 [0, 1] → GPU [1, 3, H, W]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img_t = img_t / 255.0 + 1e-8  # [1, 3, H, W] in (0, 1]

    N = len(sigma_list)

    # Multi-Scale Retinex: sum(log(I) - log(I * G_n)) / N
    log_img = torch.log10(img_t)
    msr = torch.zeros_like(log_img)

    for sigma in sigma_list:
        blurred = _separable_gaussian_blur(img_t, sigma)
        msr += log_img - torch.log10(blurred + 1e-8)

    msr /= N

    # Color Restoration: beta * (log(alpha * I) - log(sum_channels))
    ch_sum = img_t.sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cr = beta * (torch.log10(alpha * img_t + 1e-8) - torch.log10(ch_sum + 1e-8))

    result = cr * msr  # [1, 3, H, W]

    # Auto-level to [0, 255]
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    out = np.zeros_like(result_np, dtype=np.uint8)
    for c in range(3):
        ch = result_np[:, :, c]
        lo = np.percentile(ch, 1)
        hi = np.percentile(ch, 99)
        ch = np.clip((ch - lo) / (hi - lo + 1e-8) * 255.0, 0, 255)
        out[:, :, c] = ch.astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _msrcr_cpu(image, sigma_list, alpha, beta):
    """CPU fallback using OpenCV."""
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
    """
    Batch MSRCR-enhance all .jpg images in input_dir (GPU-accelerated).

    Returns:
        dict with image_count, time_elapsed_sec, avg_ms_per_image.
    """
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.jpg"))
    if not image_files:
        logger.warning("No .jpg images in %s", input_dir)
        return {"image_count": 0, "time_elapsed_sec": 0, "avg_ms_per_image": 0}

    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info("MSRCR (%s): processing %d images ...", device_str, len(image_files))
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
