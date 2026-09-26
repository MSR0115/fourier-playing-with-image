"""
2D Gerchberg-Saxton hologram retrieval -- a picture, not just 8 numbers.

The Holography page's original tool (epicycle/holography.py) runs the
Gerchberg-Saxton loop on 8 samples in 1-D, small enough to watch every wave by
eye. That is deliberately a toy model, and 8 numbers can never *look like* a
photo -- which is exactly the gap this module fills for the two newer tools
(Animated hologram, Contour -> Hologram): the same bounce-the-wavefront loop,
run with numpy.fft.fft2 / ifft2 over a real SIZE x SIZE amplitude image, so the
target, the computed hologram, and the reconstruction are all actual pictures --
the Holography-page equivalent of the Original / spectrum / filtered /
reconstructed image grid Photography's Frequency filtering panel already shows.

The physics is unchanged from the 1-D version, just in two dimensions:
  1. Illuminate the hologram with a uniform-amplitude, phase-only wave.
  2. Propagate to the image plane (fft2). Its amplitude is the reconstruction.
  3. Keep that phase, but swap in the *target* image as the amplitude there --
     this is the one place the target's own pixels enter the loop.
  4. Propagate back (ifft2), drop the amplitude that comes back, and keep only
     the phase -- that phase-only pattern is the next committed hologram.
"""
import base64

import cv2
import numpy as np

SIZE = 96   # working resolution: recognisable as a picture, fast enough to step through interactively


# ---------------------------------------------------------------- image <-> PNG data URL helpers
def _to_uint8(a):
    a = np.asarray(a, dtype=np.float64)
    lo, m = a.min(), a.max()
    a = (a - lo) / (m - lo) if m - lo > 1e-9 else np.zeros_like(a)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _png_data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def amp_to_data_url(amp):
    """A reconstruction / target amplitude array -> a grayscale PNG data URL."""
    return _png_data_url(_to_uint8(amp))


def phase_to_data_url(phase):
    """A hologram phase array (radians, -pi..pi) -> a hue-mapped PNG data URL --
    the classic "rainbow" way phase-only holograms and SLM patterns are shown,
    since phase alone has no natural brightness of its own."""
    hue = ((np.asarray(phase) + np.pi) / (2 * np.pi) * 179.0).astype(np.uint8)  # OpenCV hue is 0..179
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], axis=-1)
    return _png_data_url(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


# ---------------------------------------------------------------- targets from an uploaded photo
def _letterbox_square(gray):
    h, w = gray.shape
    side = max(h, w)
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = gray
    return canvas


def target_from_image_bytes(file_bytes, mode="edges", size=SIZE, low_thresh=60.0, high_thresh=160.0):
    """Build a size x size amplitude target from an uploaded photo.
    mode="edges": a Canny edge map (bright outline on a dark field) -- the
      Contour -> Hologram tool's link back to the Photography hub's contour and
      edge-detector ideas.
    mode="gray":  the plain grayscale photo, letterboxed square, so any photo can
      simply be "beamed" as a hologram target on its own."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read that image.")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    square = cv2.resize(_letterbox_square(gray), (size, size), interpolation=cv2.INTER_AREA)

    if mode == "edges":
        edges = cv2.Canny(cv2.GaussianBlur(square, (3, 3), 0), float(low_thresh), float(high_thresh))
        amp = edges.astype(np.float64)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
    elif mode == "gray":
        amp = square.astype(np.float64)
        contour_count = None
    else:
        raise ValueError("mode must be 'edges' or 'gray'.")

    amp = amp / (amp.max() if amp.max() > 0 else 1.0)
    amp = np.clip(amp, 0.02, 1.0)   # a hard zero anywhere makes that pixel's phase meaningless / noisy
    return {
        "amp": amp,
        "preview": amp_to_data_url(amp),
        "contourCount": contour_count,
        "sourceW": int(bgr.shape[1]),
        "sourceH": int(bgr.shape[0]),
    }


# ---------------------------------------------------------------- the GS loop itself
def random_phase(size=SIZE):
    return np.random.uniform(-np.pi, np.pi, size=(size, size))


def gs_step(hologram_phase, target_amp):
    """One Gerchberg-Saxton pass.
    Returns: (reconstructed_amp, new_hologram_phase, rmse)
      reconstructed_amp -- what the *current* (incoming) hologram_phase actually
        produces at the image plane, normalized to [0, 1] -- this is what a
        viewer would see right now, before the target-amplitude swap.
      new_hologram_phase -- the next committed phase-only hologram.
      rmse -- error between reconstructed_amp and the (normalized) target,
        computed on that same pre-swap reconstruction, so it reflects how good
        *this* hologram already is.
    """
    field_h = np.exp(1j * hologram_phase)                      # uniform illumination, unit amplitude
    field_i = np.fft.fftshift(np.fft.fft2(field_h))
    recon_amp = np.abs(field_i)
    recon_n = recon_amp / (recon_amp.max() if recon_amp.max() > 1e-9 else 1.0)
    target_n = target_amp / (target_amp.max() if target_amp.max() > 1e-9 else 1.0)
    rmse = float(np.sqrt(np.mean((recon_n - target_n) ** 2)))

    field_i_new = target_amp * np.exp(1j * np.angle(field_i))  # keep phase, swap in the target's amplitude
    field_h_new = np.fft.ifft2(np.fft.ifftshift(field_i_new))
    new_phase = np.angle(field_h_new)                          # drop amplitude, keep phase (illumination constraint)
    return recon_n, new_phase, rmse