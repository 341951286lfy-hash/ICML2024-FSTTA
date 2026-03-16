import hashlib
import math
from typing import Optional

import cv2
import numpy as np


NAVTRUST_RGB_CORRUPTIONS = [
    "none",
    "motion_blur",
    "low_light",
    "low_light_noise",
    "spatter",
    "flare",
    "defocus",
    "foreign_object",
    "black_out",
]


def make_deterministic_rng(key: str, base_seed: int = 0) -> np.random.Generator:
    """
    为每个 (scan, viewpoint, viewidx, corruption) 生成稳定随机数。
    这样离线提特征可复现。
    """
    digest = hashlib.sha256(f"{key}|{base_seed}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _motion_blur(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    img_f = _to_float(img)
    k = _odd(max(3, int(3 + 18 * severity)))
    angle = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    blurred = cv2.filter2D(img_f, -1, kernel)
    alpha = 0.55 + 0.30 * severity
    out = alpha * blurred + (1.0 - alpha) * img_f
    return _clip_uint8(out)


def _low_light_mask(h: int, w: int, severity: float, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 模拟室内局部光源：亮源中心随机，边缘更暗
    cx = float(rng.uniform(0.2 * w, 0.8 * w))
    cy = float(rng.uniform(0.1 * h, 0.7 * h))

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = dist / (np.sqrt(h * h + w * w) + 1e-6)

    base_dark = 1.0 - 0.68 * severity
    radial = 1.0 - 0.90 * severity * dist
    mask = np.clip(base_dark + (1.0 - base_dark) * radial, 0.08, 1.0)
    return mask.astype(np.float32)


def _low_light(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    img_f = _to_float(img)
    h, w = img.shape[:2]
    mask = _low_light_mask(h, w, severity, rng)
    out = img_f * mask[..., None]
    return _clip_uint8(out)


def _low_light_noise(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    dark = _low_light(img, severity, rng).astype(np.float32) / 255.0
    h, w = dark.shape[:2]

    # 简化版 CMOS 低照噪声：shot noise + read noise + row noise + quantization
    photon_scale = max(8.0, 48.0 - 28.0 * severity)
    shot = rng.poisson(np.clip(dark, 0.0, 1.0) * photon_scale) / photon_scale

    read_std = 0.01 + 0.03 * severity
    read_noise = rng.normal(0.0, read_std, size=dark.shape).astype(np.float32)

    row_std = 0.005 + 0.02 * severity
    row_noise = rng.normal(0.0, row_std, size=(h, 1, 1)).astype(np.float32)

    out = shot + read_noise + row_noise
    out = np.clip(out, 0.0, 1.0)

    # 量化噪声
    levels = max(16, int(256 - 120 * severity))
    out = np.round(out * levels) / levels

    return _clip_uint8(out * 255.0)


def _spatter(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    img_f = _to_float(img)
    h, w = img.shape[:2]

    noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    sigma = 8.0 + 18.0 * severity
    blobs = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)

    thr = np.percentile(blobs, 95.0 - 8.0 * severity)
    blobs = (blobs > thr).astype(np.float32)
    blobs = cv2.GaussianBlur(blobs, (0, 0), sigmaX=2.0 + 4.0 * severity, sigmaY=2.0 + 4.0 * severity)

    alpha = np.clip(blobs * (0.35 + 0.45 * severity), 0.0, 0.85)[..., None]
    tint = np.array([220.0, 220.0, 220.0], dtype=np.float32)[None, None, :]
    out = img_f * (1.0 - alpha) + tint * alpha
    return _clip_uint8(out)


def _flare(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    img_f = _to_float(img)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    cx = float(rng.uniform(0.1 * w, 0.9 * w))
    cy = float(rng.uniform(0.05 * h, 0.45 * h))

    radius = (0.18 + 0.35 * severity) * math.sqrt(h * h + w * w)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    radial = np.exp(-dist2 / (2.0 * radius * radius + 1e-6))

    # 加一点条纹/散射
    slope = float(rng.uniform(-0.4, 0.4))
    line = yy - (cy + slope * (xx - cx))
    streak_sigma = (0.015 + 0.04 * severity) * h
    streak = np.exp(-(line ** 2) / (2.0 * streak_sigma * streak_sigma + 1e-6))

    alpha = np.clip((0.8 * radial + 0.25 * streak) * (0.25 + 0.65 * severity), 0.0, 0.95)[..., None]
    warm = np.array([255.0, 235.0, 180.0], dtype=np.float32)[None, None, :]
    out = img_f + alpha * warm
    return _clip_uint8(out)


def _defocus(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    img_f = _to_float(img)
    k = _odd(max(3, int(3 + 20 * severity)))
    sigma = 0.8 + 2.8 * severity
    out = cv2.GaussianBlur(img_f, (k, k), sigmaX=sigma, sigmaY=sigma)
    return _clip_uint8(out)


def _foreign_object(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]

    # 论文写的是中心黑色遮挡；这里允许轻微偏移但整体仍在中央区域
    cx = int(rng.uniform(0.42 * w, 0.58 * w))
    cy = int(rng.uniform(0.42 * h, 0.58 * h))
    radius = int((0.10 + 0.18 * severity) * min(h, w))

    cv2.circle(out, (cx, cy), radius, (0, 0, 0), thickness=-1)
    return out


def _black_out(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    p = min(0.85, 0.15 + 0.60 * severity)
    if float(rng.random()) < p:
        return np.zeros_like(img, dtype=np.uint8)
    return img.copy()


def apply_navtrust_rgb_corruption(
    img_rgb: np.ndarray,
    corruption: str,
    severity: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    输入:
        img_rgb: H x W x 3, uint8, RGB
        corruption: NavTrust 风格 corruption 名称
        severity: [0, 1]
    输出:
        H x W x 3, uint8, RGB
    """
    if corruption not in NAVTRUST_RGB_CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")

    severity = float(np.clip(severity, 0.0, 1.0))
    if corruption == "none" or severity <= 0.0:
        return img_rgb.copy()

    if rng is None:
        rng = np.random.default_rng()

    if corruption == "motion_blur":
        return _motion_blur(img_rgb, severity, rng)
    if corruption == "low_light":
        return _low_light(img_rgb, severity, rng)
    if corruption == "low_light_noise":
        return _low_light_noise(img_rgb, severity, rng)
    if corruption == "spatter":
        return _spatter(img_rgb, severity, rng)
    if corruption == "flare":
        return _flare(img_rgb, severity, rng)
    if corruption == "defocus":
        return _defocus(img_rgb, severity, rng)
    if corruption == "foreign_object":
        return _foreign_object(img_rgb, severity, rng)
    if corruption == "black_out":
        return _black_out(img_rgb, severity, rng)

    raise RuntimeError(f"Unknown corruption: {corruption}")
