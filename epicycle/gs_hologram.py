"""
gs_hologram.py — Fourier hologram generator (Gerchberg-Saxton, numpy fft2d).

Backend logic for the Fourier hologram generator tool on the Holography page.
All frequency-domain work goes through numpy's real 2-D FFT (np.fft.fft2 /
np.fft.ifft2).  Target images can be text, geometric shapes, or user-uploaded
photos.

This is a *separate* engine from holography2d.py — it uses Pearson correlation
as a convergence metric, has energy-normalised target amplitudes (Parseval's
theorem), and supports text / shape / image target sources.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Fourier transform layer
# ---------------------------------------------------------------------------

def fft2d(x: np.ndarray) -> np.ndarray:
    """Forward 2-D discrete Fourier transform (unnormalized)."""
    return np.fft.fft2(x)


def ifft2d(X: np.ndarray) -> np.ndarray:
    """Inverse 2-D discrete Fourier transform (normalized by N*M)."""
    return np.fft.ifft2(X)


def fftshift2d(x: np.ndarray) -> np.ndarray:
    """Swap quadrants so the zero frequency sits at the center."""
    return np.fft.fftshift(x)


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def image_to_target_amplitude(img: Image.Image, size: int) -> np.ndarray:
    """
    Convert a PIL image into a target amplitude array (sqrt of normalized
    brightness), energy-normalized so the loop can actually converge.

    A hologram plane of unit amplitude over an N x N grid carries total
    energy N**2; by Parseval's theorem its 2-D FFT carries energy N**4.
    The target is scaled to match that same total energy.
    """
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    brightness = np.asarray(img, dtype=np.float64) / 255.0
    amp = np.sqrt(brightness)
    energy = np.sum(amp ** 2)
    goal = float(size) ** 4
    if energy > 1e-12:
        amp *= np.sqrt(goal / energy)
    return amp


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation — scale-invariant convergence metric."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-12:
        return 0.0
    return float(max(0.0, np.sum(a * b) / denom))


# ---------------------------------------------------------------------------
# Gerchberg-Saxton loop
# ---------------------------------------------------------------------------

def gs_step(phase: np.ndarray, target_shifted: np.ndarray):
    """
    One Gerchberg-Saxton pass.

    phase            current hologram phase (radians), shape (N, N)
    target_shifted   target amplitude, already fftshifted

    Returns:
        new_phase       updated hologram phase
        recon_amp_raw   raw-order reconstructed amplitude (for display)
        corr            correlation to target
    """
    hologram_field = np.exp(1j * phase)
    image_field = fft2d(hologram_field)

    recon_amp_raw = np.abs(image_field)
    image_phase = np.angle(image_field)
    corr = correlation(recon_amp_raw, target_shifted)

    constrained_field = target_shifted * np.exp(1j * image_phase)
    back_field = ifft2d(constrained_field)

    new_phase = np.angle(back_field)
    return new_phase, recon_amp_raw, corr


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def phase_to_gray(phase: np.ndarray) -> np.ndarray:
    """Map phase in (-pi, pi] to 0-255 grayscale."""
    return ((phase + np.pi) / (2 * np.pi) * 255.0).clip(0, 255).astype(np.uint8)


def recon_to_gray(recon_amp_raw: np.ndarray) -> np.ndarray:
    """fftshift, square to intensity, normalize to 0-255."""
    shifted = fftshift2d(recon_amp_raw)
    intensity = shifted ** 2
    max_val = intensity.max()
    if max_val > 1e-12:
        intensity = intensity / max_val
    return (intensity * 255.0).clip(0, 255).astype(np.uint8)


def array_to_data_url(gray_uint8: np.ndarray) -> str:
    """Convert a uint8 grayscale array to a PNG data URL."""
    img = Image.fromarray(gray_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Shape / text target generators
# ---------------------------------------------------------------------------

def build_shape_image(shape: str, size: int) -> Image.Image:
    img = Image.new("L", (size, size), color=0)
    d = ImageDraw.Draw(img)
    cx = cy = size / 2
    if shape == "circle":
        r = size * 0.30
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif shape == "ring":
        r = size * 0.27
        w = size * 0.08
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=max(1, int(w)))
    elif shape == "cross":
        t = size * 0.15
        d.rectangle([cx - t / 2, size * 0.12, cx + t / 2, size * 0.88], fill=255)
        d.rectangle([size * 0.12, cy - t / 2, size * 0.88, cy + t / 2], fill=255)
    return img


def build_text_image(text: str, size: int) -> Image.Image:
    img = Image.new("L", (size, size), color=0)
    d = ImageDraw.Draw(img)
    text = (text or "HOLOGRAM").upper()[:16]

    font = None
    font_size = int(size * 0.30)
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    max_w = size * 0.86
    if w > max_w and hasattr(font, "size"):
        scale = max_w / w
        try:
            font = font.font_variant(size=max(8, int(font_size * scale)))
            bbox = d.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
        except Exception:
            pass
    h = bbox[3] - bbox[1]
    d.text((size / 2 - w / 2 - bbox[0], size / 2 - h / 2 - bbox[1]), text, fill=255, font=font)
    return img
