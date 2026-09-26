# Fourier Playing With Image

A landing page, two darkrooms. **Photography** now has all four of its tools wired up —
**Frequency filtering**, **Hybrid Image**, **Fourier sketching** and **Epicycle drawing**.
Epicycle drops in a face photo, finds the face, levels the eyes, cuts a head-and-shoulders
crop, finds hair / eyes / nose / lips / beard / ears as separate groups of contours, joins
them into ONE closed curve and takes its Fourier transform, and the browser draws that
curve with spinning circles (epicycles). Fourier sketching turns a photo into a pencil
sketch: a Fourier high-pass filter (numpy's 2-D FFT, masked around the zero-frequency
center) finds the edges, a threshold turns them into crisp boundaries, and a color-dodge
blend shades softly inside them. **Holography** holds one tool so far, **Hologram
retrieval**: an 8-sample, 1-D Gerchberg&ndash;Saxton loop that bounces a wavefront between
the hologram and image planes, keeping only the amplitude each plane actually knows and
carrying the phase forward, until a phase-only hologram reproduces a target amplitude
pattern (four built-in shapes, or one derived from an uploaded image).

## Run it

```bash
python -m venv .venv                     # optional
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                            # then open http://127.0.0.1:5000
```

Options: `python app.py --port 8000`, `python app.py --host 0.0.0.0` (other devices on your network).
Python 3.9+ is fine. Nothing is downloaded at run time: face and eye detection use the Haar cascades that
ship inside `opencv-python`. Photos are processed in memory and are not saved.

## Pages

| Route | What's there |
| --- | --- |
| `/` | Landing page: the title, a small live epicycle demo, and the two doors below |
| `/photography` | Tool rail with 4 entries, all wired up: Frequency filtering, Hybrid Image, Fourier sketching and Epicycle drawing |
| `/holography` | Hologram retrieval: the 8-sample 1-D Gerchberg&ndash;Saxton loop, wired up to `/api/holo/*` |

## Layout

```
app.py                  Flask server: serves static/ and the JSON API
epicycle/                the vision + maths backend
  vision.py             decode, face + eyes (OpenCV Haar), level and crop, skin model, part labels
  contours.py            feature-aware Canny, stroke tracing, Focus tool (local contours)
  fourier.py              order strokes, resample to 8192 points, NumPy FFT -> coefficients
  sample.py               the built-in hand-drawn sample face
  frequency.py            the true continuous 2D Fourier transform (CFT): filtering + hybrid images
  sketch.py                Fourier sketching: numpy 2-D FFT high-pass edges -> threshold -> pencil shading
  holography.py           1-D Gerchberg-Saxton hologram retrieval (NumPy FFT), for the Holography page
static/
  index.html              Landing page
  photography.html        Photography hub: tool rail + the 4 tool panels
  holography.html          Holography: the Hologram retrieval tool
  css/
    theme.css              shared color tokens (light/dark) and base reset, used by every page
    landing.css             landing page styling + the hero's live demo canvas
    photography.css         hub chrome: header, tool rail, panels
    epicycle.css            the Epicycle drawing tool itself (was style.css) — untouched logic
    frequency.css            the Frequency filtering, Hybrid Image and Fourier sketching panels
    holography.css           the Hologram retrieval tool: controls, charts, phasors, phase history
  js/
    theme.js                light/dark toggle, applied in <head> to avoid a flash of the wrong theme
    landing.js               the hero's rotating-circles demo (decorative, independent of the backend)
    photography.js           tab switching between the 4 tool panels
    app.js                    the epicycle engine, morphing, UI, Focus / Erase tools — unchanged
    frequency.js              Frequency filtering + Hybrid Image panels, talk to /api/process, /api/hybrid
    sketch.js                 Fourier sketching panel, talks to /api/sketch
    holography.js             the Gerchberg-Saxton loop: animation, charts, image upload, talks to /api/holo/*
requirements.txt
```

The theme (light/dark), fonts and color palette are shared across all three pages via `css/theme.css`, so
switching theme on any page and coming back keeps your choice (it's saved to `localStorage`).

## API (all JSON except the photo/image uploads)

| Route | Purpose |
| --- | --- |
| `POST /api/analyze` | multipart `file` (+ `detail`, `sigma`, `min_len`): face, levelled crop (PNG data URL), labelled strokes, pupils, session id |
| `POST /api/retrace` | `{session, detail, sigma, min_len}`: same photo, new contours |
| `POST /api/focus` | `{session, rect:[x0,y0,x1,y1], level, sigma}`: new contours inside one square |
| `POST /api/fourier` | `{strokes, parts, w, h, plain}`: one closed path -> `re`, `im` (2049 coefficients), `pen`, `part` |
| `GET /api/sample` | the sample face |
| `POST /api/process` | multipart `image` (+ filter params): single-image CFT filtering, for the Frequency filtering panel |
| `POST /api/hybrid` | multipart `image_low`, `image_high` (+ params): the Hybrid Image panel |
| `POST /api/sketch` | multipart `image` (+ `cutoff`, `hp_mode`, `thresh_method`, `manual_thresh`, `pre_smooth`, `use_shading`, `blur_sigma`): grayscale, noise-reduced, edge, threshold and final pencil-sketch images, for the Fourier sketching panel |
| `POST /api/holo/target` | `{pattern}` or `{customShape}` (8 values): starts a hologram session, returns the target amplitude and a random starting phase |
| `POST /api/holo/step` | `{session_id}`: runs one Gerchberg-Saxton pass, commits the new hologram phase, returns the reconstruction and its RMSE |
| `POST /api/holo/reset` | `{session_id}`: keeps the same target, draws a fresh random starting phase |

## How the maths works

1. Strokes are joined into one polyline (a group of parts at a time, nearest neighbour inside a group).
   Jumps between strokes are flagged "pen up" and are not inked.
2. The polyline is resampled by arc length to N = 8192 points `z[n] = x[n] + i*y[n]`.
3. `c[k] = FFT(z)[k] / N` for k = -1024 ... 1024. Circle k turns k times per cycle with radius `|c[k]|`,
   and the tip of the chain is `z(t) = sum_k c[k] * exp(2*pi*i*k*t)`.
4. The front end tapers the coefficients (Lanczos window) so the drawn line is smooth, sorts the circles
   by radius and morphs between two sets of coefficients when you change the image or the settings.

Keys on the Epicycle drawing tool: `F` Focus, `E` Erase, `Esc` leave the tool, `[` `]` square size, `Ctrl+Z` undo, `Space` pause.

## How Fourier sketching works (Photography)

1. **Grayscale**, then an edge-preserving **bilateral filter** (Noise reduction) so JPEG grain and skin
   texture don't show up as false edges once the transform is taken.
2. **Fourier edge detection** (`fourier_edge_detection` in `epicycle/sketch.py`): the 2-D FFT of the
   smoothed image, a high-pass mask around the zero-frequency (DC) center — a soft Gaussian rolloff or an
   ideal hard cutoff — then the inverse FFT back, normalized to 0..255. Low frequencies (flat skin,
   background) are suppressed; edges and fine detail (hairline, eyes, jaw) survive bright.
3. **Thresholding** — Otsu (automatic) or a manual slider turns the edge map into a clean black/white
   boundary mask.
4. **Pencil shading** — a classic color-dodge blend (invert the grayscale photo, Gaussian-blur it, divide-
   blend it back with the original), kept only inside the thresholded boundary so the rest of the sheet
   stays paper-white.

This is the same idea as Frequency filtering's high-pass filter, just via numpy's fast 2-D FFT instead of
the true continuous transform (trapezoidal integration) — every control on the panel re-runs the whole
pipeline live, so the fast transform is what keeps it responsive.

## How Hologram retrieval works (Holography)

1. `POST /api/holo/target` energy-normalizes an 8-value target amplitude (one of four built-in shapes,
   or 8 brightness samples the browser pulled from an uploaded image) and starts a session with a random
   starting phase — nothing about the target ever changes after this.
2. Each `POST /api/holo/step` runs one Gerchberg-Saxton pass, entirely in `epicycle/holography.py`:
   propagate the current phase-only hologram to the image plane (`fft1d`), swap in the target's known
   amplitude there while keeping the phase just computed, propagate back (`ifft1d`), then drop the
   amplitude that comes back and keep only the phase — that's the next committed hologram.
3. The browser only animates numbers the server already computed: it morphs each of the 8 phasors from
   its old value through the raw (pre-illumination-constraint) amplitude to the new unit-amplitude one,
   and redraws the target/reconstruction bars, the phase strip, the per-sample time waves and the RMSE
   sparkline from the response.

An uploaded image is reduced to 8 raw brightness values by plain pixel sampling in the browser (not FFT
work); the energy normalization and every iteration after that still happen in Python.