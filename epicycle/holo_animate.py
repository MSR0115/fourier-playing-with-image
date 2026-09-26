"""
Time-multiplexed / animated hologram -- now a movie of actual pictures.

The Epicycle drawing tool on the Photography page turns one photo into a curve
that moves through time. This is the holography counterpart: instead of one
static target image, the 2-D Gerchberg-Saxton loop in epicycle/holography2d.py
runs against a short *sequence* of target frames -- simple shapes drawn frame by
frame (a moving dot, a growing ring, a rotating bar, a pulsing blob) -- and a
fresh phase-only hologram is converged for each one. Playing the frames back
shows a hologram whose reconstructed *image* changes over time: the same
principle behind real time-multiplexed holographic video, just with cartoon
shapes standing in for a real video feed so it converges fast enough to build in
one request.
"""
import cv2
import numpy as np

from . import holography2d as H2D

KINDS = ("moving_dot", "growing_ring", "rotating_bar", "pulsing_blob")
SIZE = 64                 # smaller than the single-photo tool: keeps a whole sequence quick to build and send
DEFAULT_FRAMES = 16
DEFAULT_ITERS = 15


def _canvas():
    return np.zeros((SIZE, SIZE), np.uint8)


def _soften(mask):
    """Raw white-on-black shape -> a smooth [0.02, 1] amplitude image. Blurring
    the hard edge also keeps the target inside what a SIZE x SIZE FFT grid can
    actually represent, so the loop converges cleanly instead of ringing."""
    f = cv2.GaussianBlur(mask.astype(np.float64), (5, 5), 0)
    f = f / (f.max() if f.max() > 0 else 1.0)
    return np.clip(f, 0.02, 1.0)


def make_sequence(kind, n_frames=DEFAULT_FRAMES):
    """Return a list of n_frames (SIZE, SIZE) amplitude targets."""
    if kind not in KINDS:
        raise ValueError(f"Unknown animation kind '{kind}'. Choose one of {KINDS}.")
    n_frames = int(np.clip(int(n_frames), 4, 40))
    cx = cy = SIZE / 2.0
    frames = []

    if kind == "moving_dot":
        path_r = SIZE * 0.28
        for i in range(n_frames):
            ang = 2 * np.pi * i / n_frames
            c = _canvas()
            cv2.circle(c, (int(cx + path_r * np.cos(ang)), int(cy + path_r * np.sin(ang))), max(3, SIZE // 12), 255, -1)
            frames.append(c)

    elif kind == "growing_ring":
        for i in range(n_frames):
            t = i / n_frames
            r = SIZE * 0.10 + SIZE * 0.28 * (0.5 + 0.5 * np.sin(2 * np.pi * t))
            c = _canvas()
            cv2.circle(c, (int(cx), int(cy)), int(r), 255, max(2, SIZE // 24))
            frames.append(c)

    elif kind == "rotating_bar":
        for i in range(n_frames):
            ang = 360.0 * i / n_frames
            c = _canvas()
            box = cv2.boxPoints(((cx, cy), (SIZE * 0.62, SIZE * 0.10), ang)).astype(np.int32)
            cv2.fillConvexPoly(c, box, 255)
            frames.append(c)

    else:  # pulsing_blob
        for i in range(n_frames):
            t = i / n_frames
            r = SIZE * 0.10 + SIZE * 0.15 * (0.5 + 0.5 * np.sin(2 * np.pi * t))
            c = _canvas()
            cv2.circle(c, (int(cx), int(cy)), int(r), 255, -1)
            frames.append(c)

    return [_soften(f) for f in frames]


def gs_converge(target_amp, iterations=DEFAULT_ITERS, phase=None):
    """Run `iterations` committed GS passes against one target amplitude image,
    starting from `phase` (or a fresh random guess). Returns the final hologram
    phase, the final reconstruction, and the RMSE after every pass."""
    phase = np.asarray(phase) if phase is not None else H2D.random_phase(target_amp.shape[0])
    rmse_hist = []
    recon = None
    for _ in range(int(np.clip(int(iterations), 1, 200))):
        recon, phase, rmse = H2D.gs_step(phase, target_amp)
        rmse_hist.append(rmse)
    return phase, recon, rmse_hist


def build_sequence(kind, n_frames=DEFAULT_FRAMES, iterations=DEFAULT_ITERS, carry_phase=True):
    """Build the whole animated hologram: one converged phase-only hologram per
    frame of make_sequence(kind, n_frames), returned as ready-to-display images.

    carry_phase=True starts each frame's GS loop from the *previous* frame's
    finished hologram (closer to how a real time-multiplexed display keeps
    refining one running pattern as the scene changes, and it tends to converge
    faster once the first frame has settled). carry_phase=False starts every
    frame from an independent random guess instead.
    """
    targets = make_sequence(kind, n_frames)
    frames_out = []
    carried_phase = None
    for target in targets:
        start_phase = carried_phase if carry_phase else None
        phase, recon, rmse_hist = gs_converge(target, iterations, start_phase)
        frames_out.append({
            "targetImg": H2D.amp_to_data_url(target),
            "phaseImg": H2D.phase_to_data_url(phase),
            "reconImg": H2D.amp_to_data_url(recon),
            "rmse": rmse_hist[-1],
        })
        carried_phase = phase
    return frames_out