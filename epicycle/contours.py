"""
contours.py: edges -> strokes, per facial part.

* extract_part_strokes: feature-aware Canny. Every part (eyes, lips, hair ...) gets its own
  threshold, taken from that part's own edge strength, so faint eyelids count as much as bold hair.
* extract_plain_strokes: the same idea for a picture with no face.
* local_contours: the "Focus" tool: strong contours inside one square, with contrast stretched
  to that square.
"""
import cv2
import numpy as np

# part id -> (gain on the gradient, share of edge crests kept, min stroke length in px, max ink in px)
PART_PARAMS = {
    1: (0.85, 0.16, 14, 3600),   # hair
    2: (0.35, 0.07, 26, 1800),   # face (contour, folds)
    3: (1.50, 0.42, 5, 2600),    # eyes + brows
    4: (1.20, 0.38, 6, 1500),    # nose
    5: (1.40, 0.45, 6, 1500),    # lips
    6: (0.90, 0.16, 14, 2200),   # beard
    7: (1.10, 0.30, 8, 1000),    # ears
    8: (0.45, 0.04, 40, 900),    # neck and shoulders
}

_DX = (1, 0, -1, 0, 1, -1, -1, 1)
_DY = (0, 1, 0, -1, 1, 1, -1, -1)
_DL = (1, 1, 1, 1, 1.4142, 1.4142, 1.4142, 1.4142)


# ---------------------------------------------------------------- image helpers
def gray_float(rgba):
    """Alpha-composite over white, then luma."""
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgb * a + 255.0 * (1 - a)
    return (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype(np.float32)


def stretch(gray, sel=None, lo_p=1, hi_p=99, min_span=24):
    vals = gray[sel] if sel is not None else gray.ravel()
    if vals.size < 100:
        return gray
    lo, hi = np.percentile(vals, [lo_p, hi_p])
    if hi - lo < min_span:
        return gray
    return np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.float32)


def sobel(blur):
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    for a in (gx, gy):
        a[0, :] = a[-1, :] = 0
        a[:, 0] = a[:, -1] = 0
    return gx, gy, np.hypot(gx, gy)


def nms(mag, gx, gy):
    """Keep only the crest of each edge, one pixel wide."""
    ax, ay = np.abs(gx), np.abs(gy)

    def sh(dy, dx):
        return np.roll(np.roll(mag, -dy, 0), -dx, 1)
    horiz = ay <= 0.4142 * ax
    vert = ay >= 2.4142 * ax
    same = (gx > 0) == (gy > 0)
    m1 = np.where(horiz, sh(0, -1), np.where(vert, sh(-1, 0), np.where(same, sh(-1, -1), sh(-1, 1))))
    m2 = np.where(horiz, sh(0, 1), np.where(vert, sh(1, 0), np.where(same, sh(1, 1), sh(1, -1))))
    out = np.where((mag > 0) & (mag >= m1) & (mag > m2), mag, 0).astype(np.float32)
    out[:2, :] = out[-2:, :] = 0
    out[:, :2] = out[:, -2:] = 0
    return out


def hysteresis(nms_map, hi_map, lo_map):
    """Weak crests survive only when they connect to a strong one."""
    weak = ((nms_map > 0) & (nms_map >= lo_map)).astype(np.uint8)
    strong = (nms_map > 0) & (nms_map >= hi_map)
    n, cc = cv2.connectedComponents(weak, connectivity=8)
    good = np.zeros(n, bool)
    good[np.unique(cc[strong & (weak > 0)])] = True
    good[0] = False
    return good[cc]


# ---------------------------------------------------------------- tracing
def trace_strokes(edge):
    """Edge mask -> list of (n, 2) float32 arrays (x, y), walking each ridge and preferring to go straight."""
    h, w = edge.shape
    W2 = w + 2
    pad = np.zeros((h + 2, W2), np.uint8)
    pad[1:-1, 1:-1] = edge
    ed = bytearray(pad.tobytes())
    vis = bytearray(len(ed))
    off = [dy * W2 + dx for dx, dy in zip(_DX, _DY)]
    deg = cv2.filter2D(pad, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT).astype(np.int16) - pad
    starts_end = np.flatnonzero((pad.ravel() > 0) & (deg.ravel() <= 1))
    starts_all = np.flatnonzero(pad.ravel() > 0)

    def extend(i, dirx, diry):
        out = []
        while True:
            best, bs = -1, -9.0
            for d in range(8):
                n = i + off[d]
                if not ed[n] or vis[n]:
                    continue
                sc = (0.35 if d < 4 else 0.0) + (dirx * _DX[d] + diry * _DY[d]) / _DL[d]
                if sc > bs:
                    bs, best = sc, d
            if best < 0:
                break
            i += off[best]
            vis[i] = 1
            out.append(i)
            dirx = 0.6 * dirx + _DX[best] / _DL[best]
            diry = 0.6 * diry + _DY[best] / _DL[best]
            l = (dirx * dirx + diry * diry) ** 0.5 or 1.0
            dirx, diry = dirx / l, diry / l
        return out

    strokes = []

    def make(i):
        vis[i] = 1
        fwd = extend(i, 0.0, 0.0)
        dx = dy = 0.0
        if fwd:
            f = fwd[0]
            dx, dy = (i % W2) - (f % W2), (i // W2) - (f // W2)
            l = (dx * dx + dy * dy) ** 0.5 or 1.0
            dx, dy = dx / l, dy / l
        bwd = extend(i, dx, dy)
        idx = bwd[::-1] + [i] + fwd
        a = np.asarray(idx, np.int64)
        strokes.append(np.stack([a % W2 - 1, a // W2 - 1], 1).astype(np.float32))

    for i in starts_end:
        if not vis[i]:
            make(int(i))
    for i in starts_all:
        if not vis[i]:
            make(int(i))
    return strokes


def smooth_stroke(s, radius):
    """Moving average so pixel staircases become curves; the end points stay put."""
    n = len(s)
    if n < 5:
        return s
    c = np.concatenate([np.zeros((1, 2)), np.cumsum(s.astype(np.float64), 0)])
    out = np.empty_like(s)
    idx = np.arange(n)
    r = np.minimum(radius, np.minimum(idx, n - 1 - idx))
    lo, hi = idx - r, idx + r + 1
    out[:] = ((c[hi] - c[lo]) / (2 * r + 1)[:, None]).astype(np.float32)
    return out


# ---------------------------------------------------------------- feature-aware extraction
def extract_part_strokes(rgba, label, weight, sigma=1.4, detail=0.72, min_len=10):
    """Returns (strokes, parts, ink_per_part). label: uint8 part map (0 = none); weight: soft prior."""
    h, w = label.shape
    gray = stretch(gray_float(rgba), sel=label > 0)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx, gy, mag = sobel(blur)

    lab = label.copy()                                   # labels leak 4 px outward for edges hugging the silhouette
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(4):
        lab = np.where(lab == 0, cv2.dilate(lab, k3), lab)
    gain = np.zeros(9, np.float32)
    for p, (g, *_r) in PART_PARAMS.items():
        gain[p] = g
    inside = (lab > 0).astype(np.float32)
    gN = cv2.GaussianBlur(gain[lab] * inside, (0, 0), 5)  # blurred so parts blend (no ridge along zone borders)
    gD = cv2.GaussianBlur(inside, (0, 0), 5)
    mag = np.where(lab > 0, mag * (gN / np.maximum(gD, 1e-3)) * np.where(label > 0, 1.0, 0.6) * weight, 0).astype(np.float32)

    crest = nms(mag, gx, gy)
    d_scale = 0.4 + 1.2 * detail
    hi_t, lo_t = np.full(9, np.inf, np.float32), np.full(9, np.inf, np.float32)
    for p, (_g, frac, _m, _c) in PART_PARAMS.items():
        vals = crest[lab == p]
        if vals.size == 0 or vals.max() <= 0:
            continue
        eps = vals.max() * 0.04
        v = np.sort(vals[vals > eps])[::-1]
        if v.size == 0:
            continue
        need = int(v.size * min(0.85, frac * d_scale))
        hi = max(eps, float(v[min(max(need - 1, 0), v.size - 1)]))
        hi_t[p], lo_t[p] = hi, max(eps, hi * 0.4)
    edge = hysteresis(crest, hi_t[lab], lo_t[lab])

    strokes, parts = [], []
    scale = min_len / 10.0
    for s in trace_strokes(edge):
        n = len(s)
        cnt = np.bincount(lab[s[:, 1].astype(int), s[:, 0].astype(int)], minlength=9)
        cnt[0] = 0
        pid = int(cnt.argmax()) or 1
        if n < max(4, PART_PARAMS[pid][2] * scale):
            continue
        strokes.append(smooth_stroke(s, 2))
        parts.append(pid)
    return _budget(strokes, parts, {p: v[3] for p, v in PART_PARAMS.items()})


def _budget(strokes, parts, caps):
    """The longest strokes survive when a part has more ink than its budget."""
    order = sorted(range(len(strokes)), key=lambda i: -len(strokes[i]))
    spent, keep_s, keep_p = {}, [], []
    for i in order:
        p, n = parts[i], len(strokes[i])
        if spent.get(p, 0) + n > caps.get(p, 13000):
            continue
        spent[p] = spent.get(p, 0) + n
        keep_s.append(strokes[i])
        keep_p.append(p)
    return keep_s, keep_p, spent


# ---------------------------------------------------------------- a picture with no face
def extract_plain_strokes(rgba, detail=0.72, sigma=1.4, min_len=10, focus=0.3):
    h, w = rgba.shape[:2]
    gray = stretch(gray_float(rgba))
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx, gy, mag = sobel(blur)
    if focus > 0.001:                                    # soften gradients far from the centre
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xs / w - 0.5) / 0.44) ** 2 + ((ys / h - 0.5) / 0.56) ** 2)
        t = np.clip((r - 0.78) / (1.25 - 0.78), 0, 1)
        m = 1 - t * t * (3 - 2 * t)
        mag = mag * (1 - focus + focus * m)
    crest = nms(mag, gx, gy)
    mx = crest.max()
    if mx <= 0:
        return [], [], {}
    eps = mx * 0.03
    v = np.sort(crest[crest > eps])[::-1]
    frac = 0.012 + 0.30 * detail ** 1.6
    hi = max(eps, float(v[min(max(int(v.size * frac) - 1, 0), v.size - 1)]))
    edge = hysteresis(crest, np.full_like(crest, hi), np.full_like(crest, max(eps, hi * 0.38)))
    strokes = [smooth_stroke(s, 2) for s in trace_strokes(edge) if len(s) >= min_len]
    strokes.sort(key=lambda s: -len(s))
    kept, used = [], 0
    for s in strokes:
        if used + len(s) > 13000:
            continue
        kept.append(s)
        used += len(s)
    return kept, [0] * len(kept), {0: used}


# ---------------------------------------------------------------- Focus tool
def local_contours(rgba, rect, sigma=1.4, level=0.6):
    """Strong contours inside rect = (x0, y0, x1, y1). level 0..1 = how much detail to pull out."""
    H, W = rgba.shape[:2]
    x0, y0 = max(1, int(np.floor(rect[0]))), max(1, int(np.floor(rect[1])))
    x1, y1 = min(W - 1, int(np.ceil(rect[2]))), min(H - 1, int(np.ceil(rect[3])))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return []
    pad = 6
    ox, oy = max(0, x0 - pad), max(0, y0 - pad)
    ex, ey = min(W, x1 + pad), min(H, y1 + pad)
    gray = gray_float(rgba[oy:ey, ox:ex])
    inner = gray[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    lo, hi = np.percentile(inner, [2, 98])                # stretch using only what is inside the square
    if hi - lo > 6:
        gray = np.clip((gray - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.float32)
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(0.8, sigma * 0.7))
    gx, gy, mag = sobel(blur)
    crest = nms(mag, gx, gy)
    sel = np.zeros(crest.shape, bool)
    sel[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = True
    vals = crest[sel]
    if vals.size == 0 or vals.max() <= 0:
        return []
    mx = float(vals.max())
    eps = mx * 0.06
    v = np.sort(vals[vals > eps])[::-1]
    if v.size == 0:
        return []
    level = float(np.clip(level, 0, 1))
    need = int(v.size * (0.14 + 0.62 * level))
    hi_t = max(eps, float(v[min(max(need - 1, 0), v.size - 1)]))
    edge = hysteresis(crest, np.full_like(crest, hi_t), np.full_like(crest, max(eps, hi_t * 0.4)))
    out = []
    for s in trace_strokes(edge):
        s = smooth_stroke(s, 1) + np.array([ox, oy], np.float32)
        inside = (s[:, 0] >= rect[0]) & (s[:, 0] <= rect[2]) & (s[:, 1] >= rect[1]) & (s[:, 1] <= rect[3])
        # split into runs that stay inside the square
        run = []
        for pt, ok in zip(s, inside):
            if ok:
                run.append(pt)
            else:
                if len(run) >= 4:
                    out.append(np.asarray(run, np.float32))
                run = []
        if len(run) >= 4:
            out.append(np.asarray(run, np.float32))
    return out
