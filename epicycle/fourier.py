"""
fourier.py: strokes -> ONE closed path -> Fourier coefficients (the epicycles).

1. order the strokes (parts are drawn in groups, nearest neighbour inside a group)
2. resample the path by arc length into G points; jumps between strokes are flagged "pen up"
3. FFT: z[n] = x[n] + i*y[n]  ->  c[k] = FFT(z)[k] / G. Circle k spins k times per cycle with radius |c[k]|.
"""
import numpy as np

G = 8192           # samples around the closed path (power of two)
F = 1024           # highest harmonic kept: 2F + 1 rotating vectors
NF = 2 * F + 1
DRAW_ORDER = [1, 7, 2, 6, 3, 4, 5, 8]      # hair, ears, face, beard, eyes, nose, lips, neck and shoulders


def order_strokes(strokes, parts, jump_thr=2.6, part_order=None):
    """Concatenate strokes into one polyline. Returns X, Y, J (1 = pen-up jump into this point), PT, close_up."""
    S = len(strokes)
    if S == 0:
        return None
    st = np.array([s[0] for s in strokes], np.float64)
    en = np.array([s[-1] for s in strokes], np.float64)
    used = np.zeros(S, bool)
    groups = part_order if (parts is not None and part_order) else [None]
    pts, jumps, pids = [], [], []
    cur = None

    def append(k, rev, dist):
        nonlocal cur
        used[k] = True
        s = strokes[k][::-1] if rev else strokes[k]
        j = np.zeros(len(s), np.uint8)
        if cur is not None and dist > jump_thr:
            j[0] = 1
        pts.append(s.astype(np.float64))
        jumps.append(j)
        pids.append(np.full(len(s), parts[k] if parts is not None else 0, np.uint8))
        cur = s[-1].astype(np.float64)

    for g in groups:
        members = np.array([k for k in range(S) if (g is None or parts[k] == g)], int)
        if members.size == 0:
            continue
        if cur is None:                                   # start with the longest stroke of the first group
            first = members[np.argmax([len(strokes[k]) for k in members])]
            append(first, False, 0.0)
        while True:
            free = members[~used[members]]
            if free.size == 0:
                break
            d1 = np.hypot(*(st[free] - cur).T)
            d2 = np.hypot(*(en[free] - cur).T)
            i1, i2 = int(d1.argmin()), int(d2.argmin())
            if d1[i1] <= d2[i2]:
                append(int(free[i1]), False, float(d1[i1]))
            else:
                append(int(free[i2]), True, float(d2[i2]))
    P = np.vstack(pts)
    J = np.concatenate(jumps)
    PT = np.concatenate(pids)
    close_up = 1 if np.hypot(*(P[0] - cur)) > jump_thr else 0
    return P[:, 0], P[:, 1], J, PT, close_up


def resample(X, Y, J, PT, close_up, w, h, jump_weight=0.3):
    """Arc-length resampling to exactly G points in world space (longest image side spans [-1, 1])."""
    n = len(X)
    if n < 2:
        return None
    S = max(w, h) / 2.0
    nxt = (np.arange(n) + 1) % n
    up = np.empty(n, np.uint8)
    up[:-1] = J[1:]
    up[-1] = close_up
    seg = np.hypot(X[nxt] - X, Y[nxt] - Y) * np.where(up > 0, jump_weight, 1.0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if not total > 1e-6:
        return None
    s = np.arange(G) * total / G
    ptr = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, n - 1)
    span = cum[ptr + 1] - cum[ptr]
    t = np.where(span > 0, (s - cum[ptr]) / np.where(span > 0, span, 1), 0.0)
    j = nxt[ptr]
    px = ((X[ptr] + (X[j] - X[ptr]) * t) + 0.5 - w / 2) / S
    py = ((Y[ptr] + (Y[j] - Y[ptr]) * t) + 0.5 - h / 2) / S
    return px, py, up[ptr], PT[ptr]


def coefficients(px, py):
    """c[f] for f = -F..F (index f + F)."""
    c = np.fft.fft(px + 1j * py) / G
    idx = (np.arange(-F, F + 1) + G) % G
    return c[idx]


def build(strokes, parts, w, h, part_order=DRAW_ORDER, jump_thr=2.6):
    o = order_strokes(strokes, parts, jump_thr, part_order)
    if o is None:
        return None
    X, Y, J, PT, close_up = o
    rs = resample(X, Y, J, PT, close_up, w, h)
    if rs is None:
        return None
    px, py, pen, part = rs
    c = coefficients(px, py)
    drawn = float(np.hypot(np.diff(X), np.diff(Y))[J[1:] == 0].sum())
    return {"re": c.real, "im": c.imag, "pen": pen, "part": part,
            "stats": {"strokes": len(strokes), "jumps": int(J.sum()), "length": int(round(drawn))}}
