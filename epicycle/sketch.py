"""
sketch.py: Fourier sketching -- a face becomes a pencil sketch.

Pipeline: grayscale -> bilateral pre-smoothing -> Fourier high-pass edge
detection -> thresholding -> color-dodge pencil shading, confined to the
thresholded edges.

The edge step is the same idea as Frequency filtering's high-pass filter
(epicycle/frequency.py): mask the spectrum around the zero-frequency (DC)
center and inverse-transform back. Frequency filtering uses the true
continuous 2D transform (trapezoidal integration) so a slider can show any
real-valued cutoff on an arbitrary grid; this module uses numpy's 2-D FFT
instead -- much faster, which matters here since every control re-runs the
whole pipeline on the fly. Low frequencies (flat skin, background) are
suppressed by the mask; edges and fine detail (hairline, eyes, jaw) survive.
"""
import base64

import cv2
import numpy as np

MAX_DIM = 900  # keep the FFT fast; very large uploads are downscaled first


# ---------------------------------------------------------------- io helpers
def _decode(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read that image.")
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        bgr = cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return bgr


def _png_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _odd(n):
    n = max(1, int(n))
    return n + 1 if n % 2 == 0 else n


# ---------------------------------------------------------------- pipeline stages
def to_grayscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def preprocess(gray, strength=2):
    """Edge-preserving pre-smoothing (bilateral filter); strength 0 skips it,
    so noise in the frequency-domain edge map doesn't come from JPEG grain."""
    strength = int(strength)
    if strength <= 0:
        return gray
    d = _odd(strength)
    return cv2.bilateralFilter(gray, d=d, sigmaColor=strength * 15, sigmaSpace=strength * 15)


def fourier_edge_detection(gray, cutoff=22.0, mode="gaussian"):
    """FFT -> high-pass mask around DC -> inverse FFT -> normalized to 0..255."""
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    y, x = np.ogrid[:rows, :cols]
    dist2 = (y - crow) ** 2 + (x - ccol) ** 2

    if mode == "ideal":
        hp_mask = np.ones((rows, cols), np.float32)
        hp_mask[dist2 <= cutoff ** 2] = 0.0
    else:  # gaussian: a soft rolloff instead of a hard ring, so lines don't ring/wave
        hp_mask = 1.0 - np.exp(-dist2 / (2.0 * (cutoff ** 2)))

    filtered = fshift * hp_mask
    back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return cv2.normalize(back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def threshold_image(edge_img, method="otsu", manual_thresh=127):
    if method == "manual":
        _, th = cv2.threshold(edge_img, int(manual_thresh), 255, cv2.THRESH_BINARY)
    else:
        _, th = cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def pencil_shade(gray, edge_mask, blur_sigma=21.0):
    """Classic color-dodge pencil effect (invert, blur, divide-blend), kept
    only inside the thresholded edges so the rest of the sheet stays paper-white."""
    inverted = 255 - gray
    ksize = _odd(int(blur_sigma * 2))
    blurred = cv2.GaussianBlur(inverted, (ksize, ksize), blur_sigma)
    shading = cv2.divide(gray, 255 - blurred, scale=256)
    canvas = np.full_like(gray, 255, dtype=np.uint8)
    canvas[edge_mask > 0] = shading[edge_mask > 0]
    return canvas


# ---------------------------------------------------------------- entry point
def run(file_bytes, cutoff=22.0, hp_mode="gaussian", thresh_method="otsu",
        manual_thresh=127, pre_smooth=2, use_shading=True, blur_sigma=21.0):
    bgr = _decode(file_bytes)
    gray = to_grayscale(bgr)
    smoothed = preprocess(gray, pre_smooth)
    edge = fourier_edge_detection(smoothed, cutoff=cutoff, mode=hp_mode)
    thresh = threshold_image(edge, thresh_method, manual_thresh)
    final = pencil_shade(gray, thresh, blur_sigma) if use_shading else (255 - thresh)

    return {
        "gray": _png_data_url(gray),
        "smoothed": _png_data_url(smoothed),
        "edge": _png_data_url(edge),
        "threshold": _png_data_url(thresh),
        "final": _png_data_url(final),
        "w": int(bgr.shape[1]),
        "h": int(bgr.shape[0]),
    }