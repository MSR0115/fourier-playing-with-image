"""
Epicycle portrait: Python backend.

    pip install -r requirements.txt
    python app.py            ->  open http://127.0.0.1:5000

The browser (static/) shows the UI and animates the epicycles. This server does the vision and the maths:
  POST /api/analyze   photo -> face, levelled crop, feature-labelled contours
  POST /api/retrace   re-trace the same photo with new Detail / Smoothing / Clean-up
  POST /api/focus     "Focus" tool: find new contours inside one square
  POST /api/fourier   strokes -> one closed path -> Fourier coefficients (NumPy FFT)
  GET  /api/sample    the built-in sample face
  POST /api/sketch    photo -> Fourier high-pass edge detection -> threshold -> pencil shading

Holography has three tools:
  1. Hologram retrieval -- the original 8-sample 1-D Gerchberg-Saxton loop
     (/api/holo/target, /api/holo/step, /api/holo/reset)
  2. Fourier hologram generator -- text/shape/image targets through a real
     2-D GS engine (epicycle/gs_hologram.py, numpy.fft.fft2/ifft2):
     POST /api/gsgen/target, /api/gsgen/step, /api/gsgen/reset
  3. Animated hologram -- a played-back sequence of GS holograms
     POST /api/holo2d/anim/build
"""
import argparse
import base64
import threading
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from epicycle import contours as C
from epicycle import fourier as Fo
from epicycle import frequency as Freq
from epicycle import holography as Holo
from epicycle import gs_hologram as GS
from epicycle import holo_animate as Anim
from epicycle import sample as Sample
from epicycle import sketch as Sk
from epicycle import vision as V

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

# ---------------------------------------------------------------- tiny in-memory session store
_SESSIONS = OrderedDict()
_LOCK = threading.Lock()
MAX_SESSIONS = 12

# a second, equally tiny store for the Holography page's original 8-sample
# Gerchberg-Saxton loop (separate from the photo sessions above: different
# shape, different lifetime)
_HOLO_SESSIONS = OrderedDict()
_HOLO_LOCK = threading.Lock()
MAX_HOLO_SESSIONS = 24

# a third store for the Fourier hologram generator's GS sessions
_GS_SESSIONS = OrderedDict()
_GS_LOCK = threading.Lock()
MAX_GS_SESSIONS = 12


def _put(data):
    sid = uuid.uuid4().hex
    with _LOCK:
        _SESSIONS[sid] = data
        while len(_SESSIONS) > MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
    return sid


def _get(sid):
    with _LOCK:
        s = _SESSIONS.get(sid)
        if s is not None:
            _SESSIONS.move_to_end(sid)
    if s is None:
        raise KeyError("This session has expired. Please choose the photo again.")
    return s


def _holo_put(data):
    sid = uuid.uuid4().hex
    with _HOLO_LOCK:
        _HOLO_SESSIONS[sid] = data
        while len(_HOLO_SESSIONS) > MAX_HOLO_SESSIONS:
            _HOLO_SESSIONS.popitem(last=False)
    return sid


def _holo_get(sid):
    with _HOLO_LOCK:
        s = _HOLO_SESSIONS.get(sid)
        if s is not None:
            _HOLO_SESSIONS.move_to_end(sid)
    if s is None:
        raise KeyError("This hologram session has expired. Please choose a target again.")
    return s


def _gs_put(data):
    sid = uuid.uuid4().hex
    with _GS_LOCK:
        _GS_SESSIONS[sid] = data
        while len(_GS_SESSIONS) > MAX_GS_SESSIONS:
            _GS_SESSIONS.popitem(last=False)
    return sid


def _gs_get(sid):
    with _GS_LOCK:
        s = _GS_SESSIONS.get(sid)
        if s is not None:
            _GS_SESSIONS.move_to_end(sid)
    if s is None:
        raise KeyError("This hologram session has expired. Please choose a target again.")
    return s


# ---------------------------------------------------------------- helpers
def _flat(strokes):
    return [np.round(s.ravel(), 2).tolist() for s in strokes]


def _png_data_url(rgba):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def _from_flat(lst):
    return [np.asarray(s, np.float32).reshape(-1, 2) for s in lst]


def _extract(sess, detail, sigma, min_len):
    if sess["kind"] == "photo":
        S, P, _ = C.extract_part_strokes(sess["rgba"], sess["label"], sess["weight"], sigma, detail, min_len)
        return S, P
    S, P, _ = C.extract_plain_strokes(sess["rgba"], detail, sigma, min_len)
    return S, P


@app.errorhandler(Exception)
def _err(e):
    code = 404 if isinstance(e, KeyError) else 400 if isinstance(e, (ValueError, TypeError)) else 500
    if code == 500:
        app.logger.exception(e)
    return jsonify(error=str(e.args[0]) if e.args else str(e)), code


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/photography")
def photography():
    return send_from_directory(app.static_folder, "photography.html")


@app.route("/holography")
def holography():
    return send_from_directory(app.static_folder, "holography.html")


# ---------------------------------------------------------------- API
@app.post("/api/analyze")
def analyze():
    t0 = time.time()
    f = request.files.get("file")
    if f is None:
        raise ValueError("No file was sent.")
    detail = float(request.form.get("detail", 0.72))
    sigma = float(request.form.get("sigma", 1.4))
    min_len = float(request.form.get("min_len", 10))
    bgr = V.decode_image(f.read())
    face = V.detect_face(bgr)

    if face is None:                                           # no face: trace the whole picture
        h, w = bgr.shape[:2]
        m = min(1.0, 480.0 / max(h, w))
        small = cv2.resize(bgr, (max(64, round(w * m)), max(64, round(h * m))), interpolation=cv2.INTER_AREA)
        rgba = cv2.cvtColor(small, cv2.COLOR_BGR2RGBA)
        sess = {"kind": "plain", "rgba": rgba}
        S, P = _extract(sess, detail, sigma, min_len)
        sid = _put(sess)
        return jsonify(session=sid, kind="plain", w=rgba.shape[1], h=rgba.shape[0], image=_png_data_url(rgba),
                       strokes=_flat(S), parts=P, faces=0, ms=int((time.time() - t0) * 1000))

    rgba = V.crop_head(bgr, face)
    weight, mask = V.prior_field(rgba)
    label, has_beard, _ = V.label_parts(rgba, mask)
    sess = {"kind": "photo", "rgba": rgba, "label": label, "weight": weight}
    S, P = _extract(sess, detail, sigma, min_len)
    sid = _put(sess)
    pl, pr = V.crop_pupils()
    return jsonify(session=sid, kind="photo", w=rgba.shape[1], h=rgba.shape[0], image=_png_data_url(rgba),
                   strokes=_flat(S), parts=P, faces=face["count"], refined=face["refined"], has_beard=bool(has_beard),
                   pupils={"l": list(pl), "r": list(pr)}, ms=int((time.time() - t0) * 1000))


@app.post("/api/retrace")
def retrace():
    j = request.get_json(force=True)
    sess = _get(j["session"])
    S, P = _extract(sess, float(j.get("detail", 0.72)), float(j.get("sigma", 1.4)), float(j.get("min_len", 10)))
    return jsonify(strokes=_flat(S), parts=P)


@app.post("/api/focus")
def focus():
    j = request.get_json(force=True)
    sess = _get(j["session"])
    rect = [float(v) for v in j["rect"]]
    S = C.local_contours(sess["rgba"], rect, float(j.get("sigma", 1.4)), float(j.get("level", 0.6)))
    parts = []
    for s in S:
        if sess["kind"] == "photo":
            x, y = s[len(s) // 2]
            lab = sess["label"]
            parts.append(int(lab[int(np.clip(round(y), 0, lab.shape[0] - 1)), int(np.clip(round(x), 0, lab.shape[1] - 1))]) or 8)
        else:
            parts.append(0)
    return jsonify(strokes=_flat(S), parts=parts)


@app.post("/api/fourier")
def fourier():
    j = request.get_json(force=True)
    strokes = _from_flat(j["strokes"])
    plain = bool(j.get("plain"))
    parts = None if plain else [int(p) for p in j["parts"]]
    built = Fo.build(strokes, parts, float(j["w"]), float(j["h"]), None if plain else Fo.DRAW_ORDER)
    if built is None:
        raise ValueError("Not enough lines to build a curve.")
    return jsonify(re=np.round(built["re"], 7).tolist(), im=np.round(built["im"], 7).tolist(),
                   pen=_b64(built["pen"]), part=_b64(built["part"]), stats=built["stats"], G=Fo.G, F=Fo.F)


@app.get("/api/sample")
def sample():
    S, P = Sample.sample_strokes()
    return jsonify(w=Sample.W, h=Sample.H, strokes=_flat(S), parts=P)


@app.post("/api/process")
def process():
    f = request.files.get("image")
    if f is None:
        raise ValueError("No image uploaded.")
    result = Freq.process_single(
        f.read(),
        filter_type=request.form.get("filter_type", "band_pass"),
        r_low=float(request.form.get("r_low", 0)),
        r_high=float(request.form.get("r_high", 50)),
        cutoff=float(request.form.get("cutoff", 50)),
        shift_amount=float(request.form.get("shift_amount", 2.0)),
        size=request.form.get("size", 150),
    )
    return jsonify(result)


@app.post("/api/hybrid")
def hybrid():
    lo, hi = request.files.get("image_low"), request.files.get("image_high")
    if lo is None or hi is None:
        raise ValueError("Please upload both a low-frequency source and a high-frequency source image.")
    result = Freq.process_hybrid(
        lo.read(), hi.read(),
        cutoff=float(request.form.get("cutoff", 20)),
        high_gain=float(request.form.get("high_gain", 1.0)),
        size=request.form.get("size", 150),
    )
    return jsonify(result)


@app.post("/api/sketch")
def sketch():
    f = request.files.get("image")
    if f is None:
        raise ValueError("No image uploaded.")
    result = Sk.run(
        f.read(),
        cutoff=float(request.form.get("cutoff", 22)),
        hp_mode=request.form.get("hp_mode", "gaussian"),
        thresh_method=request.form.get("thresh_method", "otsu"),
        manual_thresh=int(float(request.form.get("manual_thresh", 127))),
        pre_smooth=int(float(request.form.get("pre_smooth", 2))),
        use_shading=request.form.get("use_shading", "true").lower() == "true",
        blur_sigma=float(request.form.get("blur_sigma", 21)),
    )
    return jsonify(result)


def _holo_preview(sid, sess):
    """One GS pass from the current committed phase, *without* committing it, so the
    browser can show what a Step would produce before the user asks for it."""
    reconstructed_amp, raw_amp, raw_phase, rmse = Holo.gs_step(sess["phase"], sess["target_amp"])
    return jsonify(session_id=sid, iteration=sess["iteration"],
                   targetAmp=_flat1d(sess["target_amp"]), hologramPhase=_flat1d(sess["phase"]),
                   reconstructedAmp=_flat1d(reconstructed_amp), rawAmp=_flat1d(raw_amp),
                   rawPhase=_flat1d(raw_phase), rmse=rmse)


def _flat1d(arr):
    return [float(v) for v in arr]


@app.post("/api/holo/target")
def holo_target():
    j = request.get_json(force=True) or {}
    custom = j.get("customShape")
    if custom is not None:
        if len(custom) != Holo.N:
            raise ValueError(f"customShape must have exactly {Holo.N} values.")
        target_amp = Holo.normalize_shape(custom)
    else:
        target_amp = Holo.make_pattern(j.get("pattern", "single"))
    sess = {"target_amp": target_amp, "phase": Holo.random_phase(), "iteration": 0}
    sid = _holo_put(sess)
    return _holo_preview(sid, sess)


@app.post("/api/holo/step")
def holo_step():
    j = request.get_json(force=True) or {}
    sess = _holo_get(j.get("session_id"))
    reconstructed_amp, raw_amp, raw_phase, rmse = Holo.gs_step(sess["phase"], sess["target_amp"])
    sess["phase"] = raw_phase
    sess["iteration"] += 1
    return jsonify(session_id=j.get("session_id"), iteration=sess["iteration"],
                   reconstructedAmp=_flat1d(reconstructed_amp), rawAmp=_flat1d(raw_amp),
                   rawPhase=_flat1d(raw_phase), rmse=rmse)


@app.post("/api/holo/reset")
def holo_reset():
    j = request.get_json(force=True) or {}
    sid = j.get("session_id")
    sess = _holo_get(sid)
    sess["phase"] = Holo.random_phase()
    sess["iteration"] = 0
    return _holo_preview(sid, sess)


# ---------------------------------------------------------------- Animated hologram (image-based)
@app.post("/api/holo2d/anim/build")
def holo2d_anim_build():
    """Build a whole animated hologram in one call: a short sequence of target
    shape frames (epicycle/holo_animate.make_sequence), each converged to its own
    phase-only hologram with the 2-D GS engine. There is no session to step
    through afterwards -- the browser already has every frame's images and just
    plays them back."""
    j = request.get_json(force=True) or {}
    kind = j.get("kind", "moving_dot")
    if kind not in Anim.KINDS:
        raise ValueError(f"kind must be one of {Anim.KINDS}.")
    n_frames = int(j.get("frames", Anim.DEFAULT_FRAMES))
    iterations = int(j.get("iterations", Anim.DEFAULT_ITERS))
    carry_phase = bool(j.get("carryPhase", True))
    frames = Anim.build_sequence(kind, n_frames, iterations, carry_phase)
    return jsonify(kind=kind, frames=frames)


# ---------------------------------------------------------------- Fourier hologram generator (GS Generator)
def _gs_state_payload(session_id, session):
    """Compute one preview pass from the current committed phase, without
    committing it, so the browser can show what a Step would produce."""
    _, recon_amp_raw, corr = GS.gs_step(session["phase"], session["target_shifted"])
    return {
        "session_id": session_id,
        "iteration": session["iteration"],
        "correlation": session["history"][-1] if session["history"] else corr,
        "preview_correlation": corr,
        "hologram_phase": GS.array_to_data_url(GS.phase_to_gray(session["phase"])),
        "reconstruction": GS.array_to_data_url(GS.recon_to_gray(recon_amp_raw)),
    }


@app.post("/api/gsgen/target")
def gsgen_target():
    """Create a new GS session from text, a shape, or an uploaded image."""
    mode = request.form.get("mode", "text")
    size = int(request.form.get("size", 128))
    if size not in (64, 128, 256):
        raise ValueError("size must be 64, 128, or 256.")

    if mode == "image":
        f = request.files.get("image")
        if f is None:
            raise ValueError("No image uploaded.")
        from PIL import Image as PILImage
        img = PILImage.open(f.stream)
    elif mode == "shape":
        img = GS.build_shape_image(request.form.get("shape", "circle"), size)
    else:
        img = GS.build_text_image(request.form.get("text", "HOLOGRAM"), size)

    target_amp = GS.image_to_target_amplitude(img, size)
    target_display = np.asarray(img.convert("L").resize((size, size)))

    session_id = uuid.uuid4().hex
    phase = np.random.default_rng().uniform(-np.pi, np.pi, size=(size, size))
    sess = {
        "target_shifted": GS.fftshift2d(target_amp),
        "phase": phase,
        "size": size,
        "iteration": 0,
        "history": [],
    }
    with _GS_LOCK:
        _GS_SESSIONS[session_id] = sess
        while len(_GS_SESSIONS) > MAX_GS_SESSIONS:
            _GS_SESSIONS.popitem(last=False)

    payload = _gs_state_payload(session_id, sess)
    payload["target_image"] = GS.array_to_data_url(target_display.astype(np.uint8))
    payload["size"] = size
    return jsonify(payload)


@app.post("/api/gsgen/step")
def gsgen_step():
    j = request.get_json(force=True) or {}
    session_id = j.get("session_id")
    sess = _gs_get(session_id)
    new_phase, recon_amp_raw, corr = GS.gs_step(sess["phase"], sess["target_shifted"])
    sess["phase"] = new_phase
    sess["iteration"] += 1
    sess["history"].append(corr)
    return jsonify(
        session_id=session_id,
        iteration=sess["iteration"],
        correlation=corr,
        hologram_phase=GS.array_to_data_url(GS.phase_to_gray(new_phase)),
        reconstruction=GS.array_to_data_url(GS.recon_to_gray(recon_amp_raw)),
    )


@app.post("/api/gsgen/reset")
def gsgen_reset():
    j = request.get_json(force=True) or {}
    session_id = j.get("session_id")
    sess = _gs_get(session_id)
    size = sess["size"]
    sess["phase"] = np.random.default_rng().uniform(-np.pi, np.pi, size=(size, size))
    sess["iteration"] = 0
    sess["history"] = []
    return jsonify(_gs_state_payload(session_id, sess))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Epicycle portrait server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    print(f"Open http://{a.host}:{a.port} in your browser")
    app.run(host=a.host, port=a.port, debug=False, threaded=True)