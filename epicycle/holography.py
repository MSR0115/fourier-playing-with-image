"""
holography.py: backend maths for the Holography page's tool -- an 8-sample,
1-D Gerchberg-Saxton hologram-retrieval loop. A phase-only hologram is found
by bouncing a wavefront between the hologram plane and the image plane,
keeping only the amplitude each plane actually knows and carrying the phase
forward. All frequency-domain work goes through fft1d / ifft1d, thin wrappers
around numpy's 1-D FFT (np.fft.fft / np.fft.ifft). Session bookkeeping
(one hologram guess + its target per browser tab) lives in app.py, the same
way the epicycle photo sessions do.
"""
import numpy as np

N = 8


def fft1d(x: np.ndarray) -> np.ndarray:
    """Forward 1-D discrete Fourier transform (unnormalized), hologram plane -> image plane."""
    return np.fft.fft(x)


def ifft1d(X: np.ndarray) -> np.ndarray:
    """Inverse 1-D discrete Fourier transform (normalized by N), image plane -> hologram plane."""
    return np.fft.ifft(X)


def normalize_shape(shape) -> np.ndarray:
    """Energy-normalize an arbitrary 8-value shape to sum(a**2) == N**2, the
    same convention used for the built-in patterns, so any custom target
    (e.g. one derived from an uploaded image) converges the same way."""
    shape = np.asarray(shape, dtype=np.float64)
    energy = float(np.sum(shape ** 2))
    goal = float(N ** 2)
    if energy > 1e-12:
        shape = shape * np.sqrt(goal / energy)
    return shape


def make_pattern(name: str) -> np.ndarray:
    """Build one of the four demo target-amplitude patterns, energy-normalized to sum(a**2) == N**2."""
    if name == "single":
        shape = np.array([0.25, 6.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    elif name == "double":
        shape = np.array([0.2, 4.0, 0.2, 0.2, 0.2, 4.0, 0.2, 0.2])
    elif name == "smooth":
        k = np.arange(N)
        d = np.minimum(np.abs(k - 4), N - np.abs(k - 4))
        shape = np.exp(-(d ** 2) / 4) * 4 + 0.2
    elif name == "random":
        shape = 0.3 + np.random.default_rng().uniform(size=N) * 3.5
    else:
        shape = np.full(N, 0.3)

    return normalize_shape(shape)


def gs_step(phase: np.ndarray, target_amp: np.ndarray):
    """
    Runs exactly one Gerchberg-Saxton pass.

    Returns:
        reconstructed_amp  forward-transform amplitude of the *incoming* phase
                            (i.e. "how good is this guess"), before any update
        raw_amp            hologram-plane amplitude after back-propagation,
                            before the uniform-illumination constraint is reapplied
        raw_phase          hologram-plane phase after back-propagation - this
                            is what gets committed as the next hologram phase
        rmse               root-mean-square error between reconstructed_amp
                            and target_amp
    """
    slm = np.exp(1j * phase)                 # unit-amplitude illumination
    img = fft1d(slm)                          # propagate to the image plane

    reconstructed_amp = np.abs(img)
    img_phase = np.angle(img)
    rmse = float(np.sqrt(np.mean((reconstructed_amp - target_amp) ** 2)))

    constrained = target_amp * np.exp(1j * img_phase)   # match brightness
    raw = ifft1d(constrained)                            # propagate back

    raw_amp = np.abs(raw)
    raw_phase = np.angle(raw)                            # (amplitude dropped on commit)
    return reconstructed_amp, raw_amp, raw_phase, rmse


def random_phase() -> np.ndarray:
    return np.random.default_rng().uniform(-np.pi, np.pi, size=N)