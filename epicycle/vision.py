"""
vision.py: find the face, level the eyes, crop head and shoulders, label facial parts.

Detection uses the Haar cascades that ship inside opencv-python (no model download).
Every facial proportion is expressed in units of "d", the distance between the pupils
(u to the right, v downward from the pupil line). The ratios are averages measured on
43 hand-annotated faces.
"""
import cv2
import numpy as np

# ---------------------------------------------------------------- crop geometry
CROP_WD, CROP_HD, CROP_EYE_D, CROP_H = 6.0, 6.9, 2.9, 480      # 6.0d wide, 6.9d tall, pupil line 2.9d from the top
CROP_D = CROP_H / CROP_HD                                       # pixels per d inside the crop
CROP_W = int(round(CROP_WD * CROP_D))
CHIN = 1.93                                                     # chin, in d below the pupil line

# part ids (shared with the front end)
HAIR, FACE, EYES, NOSE, LIPS, BEARD, EARS, BODY = 1, 2, 3, 4, 5, 6, 7, 8
NAMES = ["background", "hair", "face", "eyes", "nose", "lips", "beard", "ears", "body"]

_HC = cv2.data.haarcascades
_FACE_A = cv2.CascadeClassifier(_HC + "haarcascade_frontalface_alt2.xml")
_FACE_B = cv2.CascadeClassifier(_HC + "haarcascade_frontalface_default.xml")
_EYE_A = cv2.CascadeClassifier(_HC + "haarcascade_eye.xml")
_EYE_B = cv2.CascadeClassifier(_HC + "haarcascade_eye_tree_eyeglasses.xml")


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- decoding
def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> BGR uint8. Honours EXIF rotation; transparent PNGs are laid on white."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # applies the EXIF orientation
    if img is None:
        raise ValueError("That file could not be decoded as an image.")
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        raw = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if raw is not None and raw.ndim == 3 and raw.shape[2] == 4:
            a = raw[:, :, 3:4].astype(np.float32) / 255.0
            img = (raw[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


# ---------------------------------------------------------------- face + eyes
def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def _find_eyes(gray, box):
    x, y, w, h = box
    y0, y1 = y + int(h * 0.12), y + int(h * 0.66)
    roi = gray[y0:y1, x:x + w]
    if roi.size == 0:
        return None
    min_sz = max(8, int(w * 0.13))
    for cascade, nb in ((_EYE_A, 3), (_EYE_B, 3), (_EYE_A, 2)):
        eyes = cascade.detectMultiScale(roi, 1.05, nb, minSize=(min_sz, min_sz))
        cands = [(x + ex + ew / 2.0, y0 + ey + eh / 2.0, ew * eh) for ex, ey, ew, eh in eyes]
        best, best_score = None, -1
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = (cands[i], cands[j]) if cands[i][0] < cands[j][0] else (cands[j], cands[i])
                sep, dy = b[0] - a[0], abs(b[1] - a[1])
                if 0.25 * w < sep < 0.62 * w and dy < 0.16 * w and 0.22 * h < (a[1] + b[1]) / 2 - y < 0.58 * h:
                    score = a[2] + b[2] - 2000 * dy / w
                    if score > best_score:
                        best, best_score = (a[:2], b[:2]), score
        if best:
            return best
    return None


def detect_face(bgr):
    """Return the most prominent face: pupils, inter-pupil distance d, roll angle. None if no face."""
    h, w = bgr.shape[:2]
    s0 = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(bgr, (int(round(w * s0)), int(round(h * s0))), interpolation=cv2.INTER_AREA) if s0 < 1 else bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_sz = max(24, int(min(gray.shape) * 0.06))
    faces = _FACE_A.detectMultiScale(gray, 1.1, 4, minSize=(min_sz, min_sz))
    if len(faces) == 0:
        faces = _FACE_B.detectMultiScale(gray, 1.08, 3, minSize=(min_sz, min_sz))
    if len(faces) == 0:
        return None
    strict = _FACE_A.detectMultiScale(gray, 1.1, 8, minSize=(min_sz, min_sz))     # a stricter second opinion
    biggest = max(f[2] for f in faces)
    cands = []
    for f in faces:
        if f[2] < 0.5 * biggest:
            continue
        eyes = _find_eyes(gray, f)
        # a face with no visible pair of eyes must also survive the strict detector, otherwise it is probably not a face
        if eyes is None and not any(_iou(f, g) > 0.5 for g in strict):
            continue
        cands.append((f, eyes))
    if not cands:
        return None
    # most prominent = widest face; a face with two clear eyes wins over one without
    cands.sort(key=lambda c: c[0][2] * (1.35 if c[1] else 1.0), reverse=True)
    (x, y, fw, fh), eyes = cands[0]
    refined = eyes is not None
    if eyes:
        (lx, ly), (rx, ry) = eyes
    else:                                              # proportions measured against pupil positions
        d = 0.395 * fw
        lx, rx, ly = x + fw / 2 - d / 2, x + fw / 2 + d / 2, y + 0.385 * fh
        ry = ly
    dx, dy = rx - lx, ry - ly
    dist = float(np.hypot(dx, dy))
    return {
        "pupL": (lx / s0, ly / s0), "pupR": (rx / s0, ry / s0),
        "d": dist / s0, "angle": float(np.arctan2(dy, dx)),
        "count": int(len(cands)), "refined": refined,
    }


# ---------------------------------------------------------------- crop
def crop_head(bgr, face):
    """Rotate so the pupils are level, scale so d = CROP_D pixels, cut a head-and-shoulders crop.
    Returns an RGBA uint8 array; pixels outside the photo are transparent."""
    h, w = bgr.shape[:2]
    d = face["d"]
    sc = min(1.0, 2.2 * CROP_D / d)                    # shrink first (area filter) if the photo is much larger
    img = cv2.resize(bgr, (max(64, int(round(w * sc))), max(64, int(round(h * sc)))), interpolation=cv2.INTER_AREA) if sc < 1 else bgr
    (lx, ly), (rx, ry) = face["pupL"], face["pupR"]
    lx, ly, rx, ry, dd = lx * sc, ly * sc, rx * sc, ry * sc, d * sc
    k = CROP_D / dd
    cs, sn = np.cos(face["angle"]), np.sin(face["angle"])
    mx, my = (lx + rx) / 2, (ly + ry) / 2
    cx0, cy0 = CROP_W / 2, CROP_EYE_D * CROP_D
    a, c, b, dm = k * cs, k * sn, -k * sn, k * cs
    e, f = cx0 - a * mx - c * my, cy0 - b * mx - dm * my
    M = np.array([[a, c, e], [b, dm, f]], np.float32)
    src4 = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    out = cv2.warpAffine(src4, M, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)


def crop_pupils():
    cx, ey = CROP_W / 2, CROP_EYE_D * CROP_D
    return (cx - CROP_D / 2, ey), (cx + CROP_D / 2, ey)


# ---------------------------------------------------------------- prior + labels
def _uv(h, w):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return (xs - w / 2) / CROP_D, (ys - CROP_EYE_D * CROP_D) / CROP_D


def prior_field(rgba):
    """A soft head-and-shoulders prior from the face geometry alone: weight 1 on the subject, fading outward."""
    h, w = rgba.shape[:2]
    u, v = _uv(h, w)
    eh = np.sqrt((u / 1.5) ** 2 + ((v + 0.05) / 2.15) ** 2)
    pH = 1 - smoothstep(0.9, 1.3, eh)
    bw = 0.95 + np.maximum(0, v - 1.6) * 1.35
    pB = (1 - smoothstep(0.85, 1.25, np.abs(u) / bw)) * smoothstep(1.35, 1.9, v)
    p = np.maximum(pH, pB)
    weight = (0.12 + 0.88 * p).astype(np.float32)
    mask = p > 0.05
    bad = (rgba[:, :, 3] < 128).astype(np.uint8)       # no edges where the crop reaches beyond the photo
    if bad.any():
        bad = cv2.dilate(bad, np.ones((15, 15), np.uint8))
        mask &= bad == 0
        weight[bad > 0] = 0
    return weight, mask


def _lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _skin_model(lab, u, v):
    sel = np.zeros(lab.shape[:2], bool)
    for pu, pv, r in ((-0.62, 0.42, 0.15), (0.62, 0.42, 0.15), (0, 0.22, 0.07), (-0.3, -0.75, 0.12), (0.3, -0.75, 0.12)):
        sel |= (u - pu) ** 2 + (v - pv) ** 2 <= r * r
    L, A, B = lab[sel, 0], lab[sel, 1], lab[sel, 2]
    med = np.median

    def mad(x, m):
        return med(np.abs(x - m))
    mL, mA, mB = med(L), med(A), med(B)
    return dict(L=float(mL), a=float(mA), b=float(mB),
                sL=max(13.0, 1.6 * float(mad(L, mL))), sA=max(4.2, 1.6 * float(mad(A, mA))), sB=max(5.0, 1.6 * float(mad(B, mB))))


def label_parts(rgba, mask):
    """Label every pixel: hair, face, eyes, nose, lips, beard, ears, body (0 = outside the prior)."""
    h, w = rgba.shape[:2]
    u, v = _uv(h, w)
    lab = _lab(rgba[:, :, :3])
    sk = _skin_model(lab, u, v)
    dL, dA, dB = (lab[:, :, 0] - sk["L"]) / sk["sL"], (lab[:, :, 1] - sk["a"]) / sk["sA"], (lab[:, :, 2] - sk["b"]) / sk["sB"]
    skin = np.sqrt(dL * dL * 0.6 + dA * dA + dB * dB) < 2.6

    au = np.abs(u)
    eye_band = (au > 0.24) & (au < 0.82) & (v > -0.2) & (v < 0.3)
    brow_band = (au > 0.05) & (au < 0.98) & (v > -0.64) & (v < -0.18)
    nose_zone = (au < 0.36) & (v > -0.05) & (v < 0.86) & ~((au > 0.24) & (v < 0.3))
    lip_zone = (u / 0.64) ** 2 + ((v - 1.15) / 0.34) ** 2 < 1
    stache = (au < 0.7) & (v > 0.72) & (v < 0.93) & ~lip_zone
    beard_zone = ((u / 1.32) ** 2 + ((v - 0.4) / 1.72) ** 2 < 1) & (v > 0.62) & ~lip_zone & ~nose_zone
    ear_zone = (au > 1.06) & (au < 1.62) & (v > -0.45) & (v < 0.85)
    head_ell = (u / 1.7) ** 2 + ((v + 0.05) / 2.3) ** 2 < 1

    # hair colour = median of clearly-hair pixels above the forehead; a beard should resemble it
    hair_sel = mask & ~skin & (v < -0.85) & (v > -2.2) & head_ell
    if hair_sel.sum() > 500:
        hl, ha, hb = (float(np.median(lab[hair_sel, i])) for i in range(3))
        beard_like = ~skin & (0.5 * (lab[:, :, 0] - hl) ** 2 + (lab[:, :, 1] - ha) ** 2 + (lab[:, :, 2] - hb) ** 2 < 17 * 17)
    else:
        beard_like = ~skin & (lab[:, :, 0] < sk["L"] * 0.5)
    zone = mask & (beard_zone | stache) & (v < 1.95) & (au < 0.75)      # centre only: side hair is not a beard
    frac = float(beard_like[zone].mean()) if zone.any() else 0.0
    has_beard = frac > 0.18

    p = np.full((h, w), FACE, np.uint8)
    p[(v > CHIN + 0.05) | (~head_ell & (v > 0.9))] = BODY
    head = p != BODY
    p[head & ((v < -1.15) | (~skin & (v < -0.42)))] = HAIR
    p[head & ~skin & (au > 1.0) & (v < 0.75)] = HAIR                     # sideburns
    if has_beard:
        p[(beard_zone | stache) & (v < CHIN + 0.3) & beard_like] = BEARD
    p[ear_zone & skin & (p != HAIR)] = EARS
    p[nose_zone] = NOSE
    p[lip_zone] = LIPS
    p[eye_band | brow_band] = EYES
    p[~mask] = 0
    return p, has_beard, frac
