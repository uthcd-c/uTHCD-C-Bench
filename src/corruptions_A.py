#!/usr/bin/env python3
"""
Version A corruption suite, ported faithfully from Notebooks/visualize_corruption.ipynb
(the implementation that matches the paper figure and R2_C1).

Differences vs version B (eval_kfold_corruption.py):
  - stroke_thinning: binarise -> Zhang-Suen thinning -> MaxFilter re-dilation
    (here the Zhang-Suen core uses skimage.morphology.skeletonize(method='zhang'),
     which implements the same algorithm but compiled/fast).
  - gaussian_noise sigma [10,20,30,40,60]; shot [60,40,25,15,8];
    impulse [0.03,0.06,0.12,0.18,0.25]; gaussian_blur r[0.7,1.5,2.5,3.5,5.0];
    defocus_blur r[1,2,3,4,6]; elastic affine+rotate; pixelate scale-fraction
    [0.9,0.7,0.5,0.33,0.2] (correctly ordered); contrast [0.7,0.5,0.35,0.2,0.1];
    scale {128,64,32,16,8}.
"""
import numpy as np
from PIL import Image, ImageFilter
from skimage.morphology import skeletonize


def _clamp(arr):
    return np.uint8(np.clip(arr, 0, 255))


def _gaussian_noise(img, s):
    arr = np.array(img).astype(np.float32)
    sigma = [10, 20, 30, 40, 60][s - 1]
    return Image.fromarray(_clamp(arr + np.random.normal(0, sigma, arr.shape)))


def _shot_noise(img, s):
    arr = np.array(img).astype(np.float32) / 255.0
    vals = [60, 40, 25, 15, 8][s - 1]
    noisy = np.random.poisson(arr * vals) / float(vals)
    return Image.fromarray(_clamp(noisy * 255.0))


def _impulse_noise(img, s):
    arr = np.array(img).copy()
    p = [0.03, 0.06, 0.12, 0.18, 0.25][s - 1]
    mask = np.random.choice([0, 1, 2], size=arr.shape[:2], p=[1 - p, p / 2, p / 2])
    out = arr.copy()
    out[mask == 1] = 0
    out[mask == 2] = 255
    return Image.fromarray(out)


def _gaussian_blur(img, s):
    radius = [0.7, 1.5, 2.5, 3.5, 5.0][s - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _defocus_blur(img, s):
    radius = [1, 2, 3, 4, 6][s - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _stroke_thinning(img, s):
    max_kernel = 9
    k = int(round(max_kernel * (1.0 - (s - 1) / 4.0)))
    if k % 2 == 0:
        k = max(1, k - 1)
    dilate_size = max(1, k)
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb.convert("L")).astype(np.uint8)
    mean, std = arr.mean(), arr.std()
    thresh = mean - k * std
    bin_mask = (arr < thresh).astype(np.uint8)
    if bin_mask.sum() == 0:                       # fallback to mean threshold
        bin_mask = (arr < arr.mean()).astype(np.uint8)
    skeleton = skeletonize(bin_mask.astype(bool), method="zhang").astype(np.uint8)
    sk_pil = Image.fromarray((skeleton * 255).astype(np.uint8))
    dilated = np.array(sk_pil.filter(ImageFilter.MaxFilter(size=dilate_size))).astype(np.uint8)
    fg = (dilated == 255)
    fg3 = np.stack([fg, fg, fg], axis=-1) if fg.ndim == 2 else fg
    out = np.array(img_rgb).astype(np.uint8).copy()
    out[~fg3] = 255
    return Image.fromarray(out)


def _contrast(img, s):
    factor = [0.7, 0.5, 0.35, 0.2, 0.1][s - 1]
    arr = np.array(img).astype(np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    return Image.fromarray(_clamp((arr - mean) * factor + mean))


def _pixelate(img, s):
    scale = [0.9, 0.7, 0.5, 0.33, 0.2][s - 1]
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    return small.resize(img.size, Image.NEAREST)


def _elastic(img, s):
    max_shift = [1, 2, 3, 4, 6][s - 1]
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    angle = np.random.uniform(-max_shift * 2, max_shift * 2)
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy)).rotate(
        angle, resample=Image.BILINEAR)


def _scale(img, s):
    target = {1: 128, 2: 64, 3: 32, 4: 16, 5: 8}[s]
    w, h = img.size
    small = img.resize((target, target), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


_FUNCS = {
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "impulse_noise": _impulse_noise,
    "gaussian_blur": _gaussian_blur,
    "defocus_blur": _defocus_blur,
    "stroke_thinning": _stroke_thinning,
    "elastic": _elastic,
    "pixelate": _pixelate,
    "contrast": _contrast,
    "scale": _scale,
}


def apply_corruption(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    if corruption in ("clean", None):
        return img
    s = max(1, min(5, int(severity)))
    f = _FUNCS.get(corruption)
    return f(img, s) if f is not None else img
