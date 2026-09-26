"""
Contour-to-hologram bridge.

Takes a photo, finds its edges the way the Photography hub's planned "Edge
detector" / "Contour finding" tools would (Canny + cv2.findContours), and turns
the traced outline into an 8-value target amplitude for the Holography page's
Gerchberg-Saxton loop -- so a *shape* drives the hologram, not just a brightness
pattern typed in or sampled pixel-by-pixel from an upload.

The n-value profile is a simple radial signature: for each of n equally-spaced
angular sectors around the contours' combined centroid, how far out the edges
reach in that direction. It is the same "outline as a function of angle" idea
behind the epicycle drawing tool's Fourier descriptors, just reduced here to n
coarse samples instead of thousands of points and 2049 Fourier coefficients.

This module is intentionally independent of epicycle/contours.py, which is tuned
for the face/stroke pipeline (feature labels, part groups, pen-up jumps) that the
Epicycle drawing tool needs; a plain Canny edge map is all a radial amplitude
profile needs here.
"""
import base64

import cv2
import numpy as np

from . import holography as Holo

N = Holo.N
MAX_DIM = 420


def _decode(file_bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not read that image.")
    h, w = bgr.shape[:2]
    m = min(1.0, MAX_DIM / max(h, w))
    if m < 1.0:
        bgr = cv2.resize(bgr, (max(1, round(w * m)), max(1, round(h * m))), interpolation=cv2.INTER_AREA)
    return bgr


def _png_data_url(gray):
    ok, buf = cv2.imencode(".png", gray)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def contour_target(file_bytes, n=N, low_thresh=60.0, high_thresh=160.0):
    """Canny-edge the photo, find its contours, and reduce them to an n-value
    radial amplitude profile. Returns the energy-normalized target amplitude
    (ready to hand straight to a Gerchberg-Saxton session), a light-on-dark edge
    preview image, and a couple of stats for the UI."""
    bgr = _decode(file_bytes)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, float(low_thresh), float(high_thresh))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.vstack(contours).reshape(-1, 2).astype(np.float64) if contours else None

    if pts is None or len(pts) < n:
        # too little found to describe a shape: a faint, flat target rather than an error
        profile = np.full(n, 0.15)
        contour_count = 0
    else:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)                      # -pi .. pi
        rad = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        bins = np.floor((ang + np.pi) / (2 * np.pi) * n).astype(int) % n
        profile = np.zeros(n)
        for k in range(n):
            in_bin = rad[bins == k]
            profile[k] = in_bin.max() if in_bin.size else 0.0
        profile = profile / profile.max() if profile.max() > 0 else np.full(n, 0.15)
        contour_count = int(len(contours))

    target_amp = Holo.normalize_shape(profile.tolist())
    preview = _png_data_url(255 - edges)  # dark edges on a light thumbnail, easier to read at a glance

    return {
        "targetAmp": [float(v) for v in target_amp],
        "preview": preview,
        "contourCount": contour_count,
        "w": int(bgr.shape[1]),
        "h": int(bgr.shape[0]),
    }