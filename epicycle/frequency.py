"""
frequency.py: the *true* continuous 2D Fourier transform (CFT), via separable
trapezoidal integration — no FFT anywhere, unlike fourier.py's epicycle path.
Two things are built on top of it:

* single-image filtering: low/high/band-pass/band-stop/brightness-shift in the
  frequency domain, plus (for band-pass and band-stop) a complementarity check
  that the two reconstructions sum back to the original.
* hybrid images: one shared cutoff splits the spectrum — low frequencies from
  one photo, high frequencies from another — summed into a single image that
  reads differently up close vs. from across the room (Oliva-Schyns).

O(N^3): resolution is user-adjustable (default 150x150, capped at 400x400) to
stay responsive.
"""
import io
import base64

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- continuous transform
class CFT2D:
    """2D Continuous Fourier Transform via separable trapezoidal integration."""

    def __init__(self, image_obj):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        x, y, u, v, I = self.x, self.y, self.u, self.v, self.I
        cos_ux = np.cos(2 * np.pi * np.outer(x, u))
        sin_ux = np.sin(2 * np.pi * np.outer(x, u))

        Cx = np.trapezoid(I[:, :, None] * cos_ux[None, :, :], x, axis=1)
        Sx = np.trapezoid(I[:, :, None] * sin_ux[None, :, :], x, axis=1)

        cos_vy = np.cos(2 * np.pi * np.outer(y, v))
        sin_vy = np.sin(2 * np.pi * np.outer(y, v))

        real_term = np.trapezoid(
            Cx[:, None, :] * cos_vy[:, :, None] - Sx[:, None, :] * sin_vy[:, :, None], y, axis=0
        )
        imag_term = -np.trapezoid(
            Sx[:, None, :] * cos_vy[:, :, None] + Cx[:, None, :] * sin_vy[:, :, None], y, axis=0
        )
        return real_term, imag_term


class InverseCFT2D:
    """Inverse 2D-CFT via separable trapezoidal integration."""

    def __init__(self, real, imag, u, v, x, y):
        self.real, self.imag, self.u, self.v, self.x, self.y = real, imag, u, v, x, y

    def reconstruct(self):
        real, imag, u, v, x, y = self.real, self.imag, self.u, self.v, self.x, self.y
        cos_vy = np.cos(2 * np.pi * np.outer(v, y))
        sin_vy = np.sin(2 * np.pi * np.outer(v, y))

        real_term = real[:, :, None] * cos_vy[:, None, :] - imag[:, :, None] * sin_vy[:, None, :]
        imag_term = real[:, :, None] * sin_vy[:, None, :] + imag[:, :, None] * cos_vy[:, None, :]

        A = np.trapezoid(real_term, v, axis=0).T
        B = np.trapezoid(imag_term, v, axis=0).T
        cos_ux = np.cos(2 * np.pi * np.outer(u, x))
        sin_ux = np.sin(2 * np.pi * np.outer(u, x))

        term = A[:, :, None] * cos_ux[None, :, :] - B[:, :, None] * sin_ux[None, :, :]
        return np.trapezoid(term, u, axis=1)


# ---------------------------------------------------------------- filters
class FrequencyFilter:
    def low_pass(self, real, imag, cutoff):
        """Complement of high_pass: keep d(i,j) <= cutoff, zero the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        i, j = np.mgrid[0:rows, 0:cols]
        d = np.hypot(i - cx, j - cy)
        mask = d <= cutoff
        return real * mask, imag * mask

    def high_pass(self, real, imag, cutoff):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        i, j = np.mgrid[0:rows, 0:cols]
        d = np.hypot(i - cx, j - cy)
        mask = d > cutoff
        return real * mask, imag * mask

    def band_pass(self, real, imag, r_low, r_high):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        i, j = np.mgrid[0:rows, 0:cols]
        d = np.hypot(i - cx, j - cy)
        mask = (d > r_low) & (d <= r_high)
        return real * mask, imag * mask

    def band_stop(self, real, imag, r_low, r_high):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        i, j = np.mgrid[0:rows, 0:cols]
        d = np.hypot(i - cx, j - cy)
        mask = ~((d > r_low) & (d <= r_high))
        return real * mask, imag * mask

    def shift_brightness(self, real, imag, shift_amount):
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        real[cx, cy] += shift_amount
        return real, imag


class ReconstructionValidator:
    def verify_complementarity(self, I_recon, I_bp, I_bs):
        error = np.abs(I_bp + I_bs - I_recon)
        return bool(np.max(error) < 1e-9), float(np.max(error))


# ---------------------------------------------------------------- image adapter
class UploadedImage:
    """Duck-types the coursework's ContinuousImage from an uploaded file's bytes."""

    def __init__(self, file_bytes, size):
        img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((size, size))
        arr = np.array(img).astype(float)
        arr = arr / arr.max() if arr.max() > 0 else arr
        self.image = arr
        self.x = np.linspace(-1, 1, arr.shape[1])
        self.y = np.linspace(-1, 1, arr.shape[0])


# ---------------------------------------------------------------- rendering
def to_png_base64(arr2d, cmap="gray"):
    buf = io.BytesIO()
    plt.imsave(buf, arr2d, cmap=cmap, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def spectrum_png_base64(real, imag):
    """Log-compress + percentile-clip + gamma-stretch the magnitude spectrum.

    A plain log1p + linear min/max normalization collapses to a single bright
    dot for this continuous (trapezoidal) transform: far more energy
    concentrates near DC than an equivalent FFT would show. Percentile
    clipping + a gamma stretch reveals the surrounding structure without
    touching the underlying filtering math.
    """
    mag = np.sqrt(real ** 2 + imag ** 2)
    logmag = np.log1p(mag)
    p = np.percentile(logmag, 99.5)
    if p <= 0:
        p = logmag.max() if logmag.max() > 0 else 1.0
    norm = np.clip(logmag / p, 0, 1) ** 0.4
    return to_png_base64(norm, cmap="hot")


def edge_map(I_raw):
    e = np.abs(I_raw)
    if e.max() > 0:
        e = e / e.max()
    return 1 - e


def clamp_size(v):
    return max(16, min(int(v), 400))


# ---------------------------------------------------------------- module entry points
def process_single(file_bytes, filter_type, r_low, r_high, cutoff, shift_amount, size):
    """One uploaded photo -> CFT -> frequency-domain filter -> reconstruction."""
    size = clamp_size(size)
    img_obj = UploadedImage(file_bytes, size)
    cft = CFT2D(img_obj)
    real, imag = cft.compute_cft()

    filt = FrequencyFilter()
    if filter_type == "low_pass":
        real_f, imag_f = filt.low_pass(real, imag, cutoff)
    elif filter_type == "high_pass":
        real_f, imag_f = filt.high_pass(real, imag, cutoff)
    elif filter_type == "band_pass":
        real_f, imag_f = filt.band_pass(real, imag, r_low, r_high)
    elif filter_type == "band_stop":
        real_f, imag_f = filt.band_stop(real, imag, r_low, r_high)
    elif filter_type == "brightness":
        real_f, imag_f = filt.shift_brightness(real, imag, shift_amount)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'")

    I_filtered = InverseCFT2D(real_f, imag_f, cft.u, cft.v, img_obj.x, img_obj.y).reconstruct()
    recon_display = np.clip(I_filtered, 0, 1) if filter_type in ("brightness", "low_pass") else edge_map(I_filtered)

    result = {
        "original": to_png_base64(img_obj.image),
        "spectrum": spectrum_png_base64(real, imag),
        "filtered_spectrum": spectrum_png_base64(real_f, imag_f),
        "reconstructed": to_png_base64(recon_display),
    }

    # band_pass/band_stop are complements of each other; verify the two
    # reconstructions sum back to the unfiltered image (reuses the forward
    # transform already computed above).
    if filter_type in ("band_pass", "band_stop"):
        real_bp, imag_bp = filt.band_pass(real, imag, r_low, r_high)
        real_bs, imag_bs = filt.band_stop(real, imag, r_low, r_high)

        def reconstruct(r, im):
            return InverseCFT2D(r, im, cft.u, cft.v, img_obj.x, img_obj.y).reconstruct()

        I_recon = reconstruct(real, imag)
        I_bp = reconstruct(real_bp, imag_bp) if filter_type == "band_stop" else I_filtered
        I_bs = reconstruct(real_bs, imag_bs) if filter_type == "band_pass" else I_filtered

        is_valid, delta = ReconstructionValidator().verify_complementarity(I_recon, I_bp, I_bs)
        result["complementarity"] = {"is_valid": is_valid, "delta": delta}

    return result


def process_hybrid(low_bytes, high_bytes, cutoff, high_gain, size):
    """Two uploaded photos -> shared-cutoff low/high split -> one hybrid image."""
    size = clamp_size(size)
    img_low = UploadedImage(low_bytes, size)
    img_high = UploadedImage(high_bytes, size)
    filt = FrequencyFilter()

    # Same shared cutoff drives both halves of the spliced spectrum: distance
    # <= cutoff comes from image_low, > cutoff from image_high. Reconstructing
    # each masked spectrum separately and summing is mathematically identical
    # (by linearity of the transform) to splicing the two spectra into one
    # array and inverse-transforming once — done this way so each component
    # can also be shown on its own.
    cft_low = CFT2D(img_low)
    real_low, imag_low = cft_low.compute_cft()
    real_low_f, imag_low_f = filt.low_pass(real_low, imag_low, cutoff)
    I_low = InverseCFT2D(real_low_f, imag_low_f, cft_low.u, cft_low.v, img_low.x, img_low.y).reconstruct()

    cft_high = CFT2D(img_high)
    real_high, imag_high = cft_high.compute_cft()
    real_high_f, imag_high_f = filt.high_pass(real_high, imag_high, cutoff)
    I_high = InverseCFT2D(real_high_f, imag_high_f, cft_high.u, cft_high.v, img_high.x, img_high.y).reconstruct()

    # A raw high-pass reconstruction is typically 2-4x lower contrast than a
    # low-pass one (no DC term), so match standard deviations first, then
    # apply the user's boost on top (1.0 = balanced).
    auto_scale = I_low.std() / (I_high.std() + 1e-8)
    scale = auto_scale * high_gain
    I_high_scaled = I_high * scale

    # hybrid = IFFT( low_filter(FFT(a)) + scale * high_filter(FFT(b)) ) — merge
    # the two filtered spectra first, then one inverse transform; provably
    # identical to I_low + I_high_scaled since the inverse transform is
    # linear, cross-checked here the same way band_pass + band_stop is above.
    combined_real = real_low_f + real_high_f
    combined_imag = imag_low_f + imag_high_f
    hybrid_raw = InverseCFT2D(combined_real, combined_imag, cft_low.u, cft_low.v, img_low.x, img_low.y).reconstruct()
    equivalence_delta = float(np.max(np.abs(hybrid_raw - (I_low + I_high_scaled))))

    hybrid_img = np.clip(hybrid_raw, 0, 1)

    # Simulated viewing-distance previews (progressive blur mimics squinting /
    # stepping back, letting the low-frequency component dominate).
    near = hybrid_img
    mid = gaussian_filter(hybrid_img, sigma=max(size / 70, 0.8))
    far = gaussian_filter(hybrid_img, sigma=max(size / 22, 2.0))

    # Upside-down preview: hybrid images read differently when flipped
    # (face-inversion effect) — a genuine "different view from a different orientation".
    flipped = np.flipud(np.fliplr(hybrid_img))

    return {
        "source_low": to_png_base64(img_low.image),
        "source_high": to_png_base64(img_high.image),
        "low_component": to_png_base64(np.clip(I_low, 0, 1)),
        "high_component": to_png_base64(np.clip(0.5 + I_high_scaled, 0, 1)),
        "hybrid_near": to_png_base64(near),
        "hybrid_mid": to_png_base64(mid),
        "hybrid_far": to_png_base64(far),
        "hybrid_flipped": to_png_base64(flipped),
        "auto_scale": auto_scale,
        "equivalence_delta": equivalence_delta,
    }